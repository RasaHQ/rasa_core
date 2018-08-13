from rasa_core.policies.forms import SimpleForm
from rasa_core.events import SlotSet, StartForm, EndForm
from rasa_core.actions import ActionStartForm, Action, ActionEndForm


class TestForm(SimpleForm):
    def __init__(self):
        name = 'restaurant_form'
        slot_dict = {"people": {"ask_utt": "utter_ask_people"},
                     "location": {"ask_utt": "utter_ask_location"}}

        finish_action = "deactivate_form"

        exit_dict = {"goodbye": "deactivate_form",
                     "request_hotel": "deactivate_form_switch"}

        chitchat_dict = {"chitchat": "utter_chitchat"}

        details_intent = "utter_ask_details"

        rules = {"cuisine":{"mcdonalds": {'need':['location'], 'lose':['people', 'price']}}}

        super(TestForm, self).__init__(name, slot_dict, finish_action, exit_dict, chitchat_dict, details_intent, rules)


class StartTestForm(ActionStartForm):

    def __init__(self):
        self._name = 'activate_restaurant'

    def run(self, dispatcher, tracker, domain):
        """Simple run implementation uttering a (hopefully defined) template."""
        return [StartForm(domain, 'restaurant_form')]

    def name(self):
        return self._name

    def __str__(self):
        return "ActivateForm('{}')".format(self.name())


class StopForm(ActionEndForm):
    def __init__(self):
        self._name = 'deactivate_form'

    def run(self, dispatcher, tracker, domain):
        unfilled = tracker.active_form.check_unfilled_slots(tracker)
        if len(unfilled) == 0:
            complete = True
        else:
            complete = False
        return [EndForm(), SlotSet('form_complete', complete)]

    def name(self):
        return self._name

    def __str__(self):
        return "StopFormSwitch('{}')".format(self.name())
