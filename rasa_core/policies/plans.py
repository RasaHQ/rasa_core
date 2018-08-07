
# from rasa_core.trackers import DialogueStateTracker
# from rasa_core.domain import Domain
import numpy as np
from rasa_core.actions import Action
from rasa_core.events import Event, SlotSet
import logging


logger = logging.getLogger(__name__)



class Plan(object):
    """Next action to be taken in response to a dialogue state."""

    def next_action_idx(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[Event]
        """
        Choose an action idx given the current state of the tracker and the plan.

        Args:
            tracker (DialogueStateTracker): the state tracker for the current user.
                You can access slot values using ``tracker.get_slot(slot_name)``
                and the most recent user message is ``tracker.latest_message.text``.
            domain (Domain): the bot's domain

        Returns:
            idx: the index of the next planned action in the domain

        """

        raise NotImplementedError

    def __str__(self):
        return "Plan('{}')".format(self.name)


class SimpleForm(Plan):
    def __init__(self, name, slot_dict, finish_action, exit_dict=None, chitchat_dict=None, details_intent=None, rules=None):
        self.name = name
        self.slot_dict = slot_dict

        self.required_slots = list(self.slot_dict.keys())
        # exit dict is {exit_intent_name: exit_action_name}
        self.exit_dict = exit_dict
        self.chitchat_dict = chitchat_dict
        self.finish_action = finish_action
        self.details_intent = details_intent
        self._validate_slots()
        self.rules_yaml = rules
        self.rules = self._process_rules(self.rules_yaml)

        self.last_question = None
        self.queue = []
        self.current_required = self.required_slots

    def _validate_slots(self):
        for slot, values in self.slot_dict.items():
            if 'ask_utt' not in list(values.keys()):
                logger.error('ask_utt not found for {} in plan {}. An utterance is required to ask for a certain slot'.format(slot, self.name))
            if 'clarify_utt' not in list(values.keys()) and self.details_intent not in [None, []]:
                logger.warning('clarify_utt not found for {} in plan {}, even though {} is listed as a details intent.'.format(slot, self.name, self.details_intent))

    @staticmethod
    def _process_rules(rules):
        rule_dict = {}
        for slot, values in rules.items():
            for value, rules in values.items():
                rule_dict[(slot, value)] = (rules.get('need'), rules.get('lose'))
        return rule_dict

    def _update_requirements(self, tracker):
        #type: (DialogueStateTracker)
        if self.rules is None:
            return
        all_add, all_take = [], []
        for slot_tuple in list(tracker.current_slot_values().items()):
            if slot_tuple in self.rules.keys():
                add, take = self.rules[slot_tuple]
                if add is not None:
                    all_add.extend(add)
                if take is not None:
                    all_take.extend(take)
        self.current_required = self._prioritise_questions(list(set(self.required_slots+all_add)-set(all_take)))

    def _prioritise_questions(self, slots):
        return sorted(slots, key=lambda l: self.slot_dict[l].get('priority', 1E5))

    def check_unfilled_slots(self, tracker):
        current_filled_slots = [key for key, value in tracker.current_slot_values().items() if value is not None]
        still_to_ask = list(set(self.current_required) - set(current_filled_slots))
        still_to_ask = self._prioritise_questions(still_to_ask)
        return still_to_ask

    def _run_through_queue(self, domain):
        if self.queue == []:
            return None
        else:
            return domain.index_for_action(self.queue.pop(0))

    def _question_queue(self, question):
        queue = [self.slot_dict[question]['ask_utt'], 'action_listen']
        if 'follow_up_action' in self.slot_dict[self.last_question].keys():
            queue.append(self.slot_dict[self.last_question]['follow_up_action'])
        return queue

    def _details_queue(self, intent, tracker):
        # details will perform the clarify utterance and then ask the question again
        self.queue = [self.slot_dict[self.last_question]['clarify_utt']]
        self.queue.extend(self._question_queue(self.last_question))

    def _chitchat_queue(self, intent, tracker):
        # chitchat queue will perform the chitchat action and return to the last question
        self.queue = [self.chitchat_dict[intent]]
        self.queue.extend(self._question_queue(self.last_question))

    def _exit_queue(self, intent, tracker):
        # If the exit dict is called, the plan will be deactivated
        self.queue = [self.exit_dict[intent]]

    def _decide_next_question(self, still_to_ask, tracker):
        return still_to_ask[0]

    def next_action_idx(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> int

        out = self._run_through_queue(domain)
        if out is not None:
            # There are still actions in the queue
            return out

        intent = tracker.latest_message.parse_data['intent']['name'].replace('plan_', '', 1)
        self._update_requirements(tracker)

        if intent in self.exit_dict.keys():
            # actions in this dict should deactivate this plan in the tracker
            self._exit_queue(intent, tracker)
            return self._run_through_queue(domain)

        elif intent in self.chitchat_dict.keys() and tracker.latest_action_name not in self.chitchat_dict.values():
            self._chitchat_queue(intent, tracker)
            return self._run_through_queue(domain)
        elif intent == self.details_intent and tracker.latest_action_name != self.slot_dict[self.last_question].get('clarify_utt', None):
            self._details_queue(intent, tracker)
            return self._run_through_queue(domain)

        still_to_ask = self.check_unfilled_slots(tracker)

        if len(still_to_ask) == 0:
            self.queue = [self.finish_action, 'action_listen']
            return self._run_through_queue(domain)
        else:
            self.last_question = self._decide_next_question(still_to_ask, tracker)
            self.queue = self._question_queue(self.last_question)
            return self._run_through_queue(domain)

    def as_dict(self):
        return {"name": self.name,
                "required_slots": self.slot_dict,
                "finish_action": self.finish_action,
                "exit_dict": self.exit_dict,
                "chitchat_dict": self.chitchat_dict,
                "details_intent": self.details_intent,
                "rules": self.rules_yaml}
