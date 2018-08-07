from rasa_core.policies.plans import SimpleForm
from rasa_core.events import SlotSet, StartPlan, EndPlan
from rasa_core.actions import ActionStartPlan, Action


class TestPlan(SimpleForm):
    def __init__(self):
        name = 'restaurant_plan'
        slot_dict = {"people": {"ask_utt": "utter_ask_people"},
                     "location": {"ask_utt": "utter_ask_location"}}

        finish_action = "deactivate_plan"

        exit_dict = {"goodbye": "deactivate_plan",
                     "request_hotel": "deactivate_plan_switch"}

        chitchat_dict = {"chitchat": "utter_chitchat"}

        details_intent = "utter_ask_details"

        rules = {"cuisine":{"mcdonalds": {'need':['location'], 'lose':['people', 'price']}}}

        super(TestPlan, self).__init__(name, slot_dict, finish_action, exit_dict, chitchat_dict, details_intent, rules)

class StartTestPlan(ActionStartPlan):

    def __init__(self):
        self._name = 'activate_restaurant'

    def run(self, dispatcher, tracker, domain):
        """Simple run implementation uttering a (hopefully defined) template."""
        return [StartPlan(domain, 'restaurant_plan')]

    def name(self):
        return self._name

    def __str__(self):
        return "ActivatePlan('{}')".format(self.name())

class StopPlan(Action):
    def __init__(self):
        self._name = 'deactivate_plan'

    def run(self, dispatcher, tracker, domain):
        unfilled = tracker.active_plan.check_unfilled_slots(tracker)
        if len(unfilled) == 0:
            complete = True
        else:
            complete = False
        return [EndPlan(), SlotSet('plan_complete', complete)]

    def name(self):
        return self._name

    def __str__(self):
        return "StopPlanSwitch('{}')".format(self.name())
