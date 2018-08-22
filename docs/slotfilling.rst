.. _slotfilling:

Slot Filling
============

One of the most common conversation patterns
is to collect a few pieces of information
from a user in order to do something (book a restaurant, call an API, search a database, etc.).
This is also called **slot filling**.



Example: Providing the Weather
------------------------------


Let's say you are building a weather bot ⛅️. If somebody asks you for the weather, you will
need to know their location. Users might say that right away, e.g. `What's the weather in Caracas?`
When they don't provide this information, you'll have to ask them for it. 
We can provide two stories to Rasa Core, so that it can learn to handle both cases:

.. code-block:: md

    # story1
    * ask_weather{"location": "Caracas"}
       - action_weather_api

    # story2
    * ask_weather
       - utter_ask_location
    * inform{"location": "Caracas"}
       - action_weather_api

Here we are assuming you have defined an ``inform`` intent, which captures the cases where a user 
is just providing information.

But :ref:`customactions` can also set slots, and these can also influence the conversation. 
For example, a location like `San Jose` could refer to multiple places, in this case, probably in
Costa Rica 🇨🇷  or California 🇺🇸

Let's add a call to a location API to deal with this. 
Start by defining a ``location_match`` slot:

.. code-block:: md
    
    slots:
      location_match:
        type: categorical
        values:
        - zero
        - one
        - multiple


And our location api action will have to use the API response to fill in this slot.
It can ``return [SlotSet("location_match", value)]``, where ``value`` is one of ``"zero"``, ``"one"``, or 
``"multiple"``, depending on what the API sends back. 

We then define stories for each of these cases:


.. code-block:: md
    :emphasize-lines: 12-13, 18-19, 24-25

    # story1
    * ask_weather{"location": "Caracas"}
       - action_location_api
       - slot{"location_match": "one"}
       - action_weather_api

    # story2
    * ask_weather
       - utter_ask_location
    * inform{"location": "Caracas"}
       - action_location_api
       - slot{"location_match": "one"}
       - action_weather_api

    # story3
    * ask_weather{"location": "the Moon"}
       - action_location_api
       - slot{"location_match": "none"}
       - utter_location_not_found

    # story4
    * ask_weather{"location": "San Jose"}
       - action_location_api
       - slot{"location_match": "multiple"}
       - utter_ask_which_location


Now we've given Rasa Core a few examples of how to handle the different values
that the ``location_match`` slot can take.
Right now, we still only have four stories, which is not a lot of training data.
:ref:`interactive_learning` is agreat way to explore more conversations 
that aren't in your stories already.
The best way to improve your model is to test it yourself, have other people test it,
and correct the mistakes it makes. 


Debugging
~~~~~~~~~

The first thing to try is to run your bot with the ``debug`` flag, see :ref:`debugging` for details.
If you are just getting started, you probably only have a few hand-written stories.
This is a great starting point, but 
you should give your bot to people to test **as soon as possible**. One of the guiding principles
behind Rasa Core is:

.. pull-quote:: Learning from real conversations is more important than designing hypothetical ones

So don't try to cover every possiblity in your hand-written stories before giving it to testers.
Real user behavior will always surprise you! 


Slot Filling with Forms
-----------------------

An alternative to writing stories directly is to use Forms, an object which bypasses the policy ensemble and carries out strict logic for the purpose of filling slots.

This is preferable to writing stories to fill forms because:

- It doesn't require rewriting stories if the structure of the form is changed. I.e. if an extra slot is required to be filled then the flow can be changed simply by editing an object instead of rewriting stories.

- The intent is ignored except for in specific cases where the user does not want to fill out the form. (for example if they say goodbye).

- There can be conditional logic on key-value pairs. So if a certain slot is set by the questioning, the remaining slots to be filled can be altered.

- The bot has explicit handling of chitchat and asking for details which can be customized by the user.

NOTE: The Forms object is in beta, and has not undergone rigorous external testing. If you find any bugs or have any feature requests, please raise an issue in this repo.

Forms object
~~~~~~~~~~~~

The most simple format of Forms need only 4 things defined:

1. ``name``: the name of the Form

2. ``slot_dict``: a dictionary: ``{'FIRST_SLOT_NAME': {'ask_utt': 'WHICH_UTTERANCE_ASKS_FOR_SLOT'}, 'SECOND_SLOT_NAME':.. }``, which ties together slot names and utterances. The bot will continue to ask about the unfilled slots until all the slots are filled or the form is otherwise exited.

3. ``finish_action``: this is the name of the action that will be called when all of the relevant slots are filled. This action must return a ``EndForm`` event but can do anything else alongside it.

4. ``exit_dict``: the exit dict is a set of ``{'intent':'action'}`` pairs which describe what the bot should do in certain situations where the form should be exited.

This is currently defined as a python object. An example of the Form object defined in the ``rasa_pysdk/examples/formbot`` is:

.. code-block:: python

    class RestaurantForm(SimpleForm):
        def __init__(self):
            name = 'restaurant_form'
            slot_dict = {
                         "price": {
                                   "ask_utt": "utter_ask_price",
                                   "clarify_utt": "utter_explain_price_restaurant",
                                   "priority":0
                                   },
                         "cuisine": {
                                     "ask_utt": "utter_ask_cuisine",
                                     "clarify_utt": "utter_explain_cuisine_restaurant"
                                     },
                         "people": {
                                    "ask_utt": "utter_ask_people",
                                    "clarify_utt": "utter_explain_people_restaurant"
                                    },
                         "location": {
                                      "ask_utt": "utter_ask_location",
                                      "clarify_utt": "utter_explain_location_restaurant"
                                      }
                         }

            finish_action = "deactivate_form"

            exit_dict = {
                         "goodbye": "deactivate_form",
                         "request_hotel": "deactivate_form_switch"
                         }

            chitchat_dict = {"chitchat": "utter_chitchat"}

            details_intent = "utter_ask_details"

            rules = {
                     "cuisine":{
                                "mcdonalds": {
                                              'need':['location'],
                                              'lose':['people', 'price']
                                             }
                                }
                    }

            failure_action = 'utter_human_hand_off'

            super(RestaurantForm, self).__init__(name, slot_dict, finish_action,
                                                 exit_dict, chitchat_dict, details_intent,
                                                 rules, failure_action=failure_action)
The extra arguments not defined above are defined below, but are optional.

Stories
~~~~~~~

We also need to let Rasa Core predict when to activate the Forms. We do this by defining an action (in this bot example ``activate_restaurant`` and ``activate_hotel``) which contains a ``StartForm`` event. We then write stories where this action is triggered:

.. code-block:: md

    ## Generated Story 6817547858592778997
    * request_restaurant
        - activate_restaurant
        - slot{"switch": false}
        - slot{"cuisine": "mexican"}
        - slot{"form_complete": false}
        - utter_happy
    * chitchat
        - utter_chitchat
    * request_hotel
        - activate_hotel


which allows core to predict when to activate the form. Using a combination of the ``finish_action`` or ``exit_dict`` you can tell core to act differently dependent on the way that the form finished. In this case, we set a slot to say after an exit whether the form had been completed or not (``form_complete``).
It is important to note that *how* the slots were filled within the form does not get noticed by core. The only thing that matters is which slots are filled at the time of the form's deactivation and these influence downstream core predictions. We see above an example of a story where the form has not been filled and the user has exited.
 We also have a story where the slots of the form have all been filled:

.. code-block:: md

    ## Generated Story 7536939952037997255
    * request_restaurant
        - activate_restaurant
        - slot{"switch": false}
        - slot{"location": "Berlin"}
        - slot{"price": "high"}
        - slot{"cuisine": "mcdonalds"}
        - slot{"form_complete": true}
        - utter_filled_slots
        - utter_suggest_restaurant
    * affirm
        - utter_book_restaurant


We see in this case since the ``form_complete`` slot is set to true, we follow a different path when exiting.

``StartForm`` event
~~~~~~~~~~~~~~~~~~~
To tell the tracker that you need the form to take over predictions of actions, you have to pass a ``StartForm`` event.
The way we do that in the formbot example is to have an action which is predicted such as:

.. code-block:: python

    class StartFormAction(Action):
        def name(self):
            return "start_restaurant"

        def run(self, dispatcher, tracker, domain, executor):
            return [StartForm("restaurant_form")]

Once this event is passed, the policy will be informed that it shouldn't use the predictions of the normal policy but
instead should look for a form known in this case as ``"restaurant_form"`` with the name which is defined above.

Example output
^^^^^^^^^^^^^^
Here is an example of the debug log for a forms bot.

.. code-block:: bash

    Bot loaded. Type a message and press enter:
    request_restaurant
    2018-08-01 09:50:12 DEBUG    rasa_core.tracker_store  - Creating a new tracker for id 'default'.
    2018-08-01 09:50:12 DEBUG    rasa_core.processor  - Received user message 'request_restaurant' with intent '{'name': 'request_restaurant', 'confidence': 1.0}' and entities '[]'
    2018-08-01 09:50:12 DEBUG    rasa_core.policies.memoization  - There is a memorised next action '48'
    2018-08-01 09:50:12 DEBUG    rasa_core.policies.ensemble  - Predicted next action using policy_0_MemoizationPolicy
    2018-08-01 09:50:12 DEBUG    rasa_core.policies.ensemble  - Predicted next action 'activate_restaurant' with prob 1.00.
    2018-08-01 09:50:12 DEBUG    rasa_core.processor  - Action 'activate_restaurant' ended with events '['<rasa_core.events.StartForm object at 0x123fc92b0>' 'SlotSet(key: switch, value: False)', 'SlotSet(key: form_complete, value: False)']'
    2018-08-01 09:50:12 DEBUG    rasa_core.policies.ensemble  - Form restaurant_form predicted next action UtterAction('utter_ask_price')
    What price range?

The key lines to note are the rasa_core.policies.ensemble lines. The activation of the Form is predicted by the memoization policy and then the subsequent question asking is predicted by the Form. This will be the case until a StopForm object is passed again.

Optional arguments
~~~~~~~~~~~~~~~~~~

Advanced Forms object
^^^^^^^^^^^^^^^^^^^^^
There is added functionality which can be used:
1. ``name`` - as above
2. ``slot_dict``: We can augment the dictionaries we assign to our slots like so:
``slot_dict = {'FIRST_SLOT_NAME': {'ask_utt': 'WHICH_UTTERANCE_ASKS_FOR_SLOT', "clarify_utt": 'WHICH_UTTERANCE_EXPLAINS_SLOT', "follow_up_action": "WHICH_ACTION_SHOULD_BE_PERFORMED_AFTER_USER_REPLIES"}, ...}``
    - ``follow_up_action`` will be performed after the user responds to ``'ask_utt'``. This can be useful in some cases where you would like to ask a yes/no question. You can then have an action to deal with affirm/deny, such as `SpaAnswerParse` in `form_actions.py`
    - ``clarify_utt`` will be said if the user asks for clarification, with ``details_intent`` (explained below)
    - ``priority``: the lower the value of the priority, the sooner this question will be asked. i.e. if you would like a question to be asked first, set it to ``"priority":0``
3. ``finish_action``: as above
4. ``exit_dict``: as above
5. ``chitchat_dict``: another {"intent":"action"} dictionary, however in this case the bot, when detecting the relevant intent, will do the corresponding action and then repeat their original question. OPTIONAL
6. ``details_intent``: The intent which is asking for details about the previous question in the form fill. If the bot detects the details intent it will try to execute slot_dict['CURRENT_SLOT_NAME']['clarify_utt']. OPTIONAL
7. ``rules``: a dictionary, defined as ``{slot:{value:{keep:[slot,slot2], lose:[slot3]},...}, ...}`` which, when matching slot/value pairs will alter which slots need to be filled to trigger the finish action of the Form. This is implemented in the restaurant form OPTIONAL
8. ``max_turns``: the maximum number of turns without completion that the bot will do before exiting with ``failure_action``. Defaults to 10
9. ``failure_action``: action which will occur when the maximum number of turns has been passed. This defaults to the finish_action but can be set to be anything
The Forms need to be made as objects and then referenced in the domain (see domain.yml here). Core will trigger the Form when your activate action is predicted, and stories/featurizer will ignore the intents/actions carried out within the Form, with the exception of slot setting.

Advanced stories
^^^^^^^^^^^^^^^^
In the example here the slots for location/price/cuisine etc. are unfeaturized, so adding another slot within the form would not require rewriting the stories. Therefore to Rasa core the above story is equivalent to:

.. code-block:: md

    ## Generated Story 7536939952037997255
    * request_restaurant
        - activate_restaurant
        - slot{"switch": false}
        - slot{"form_complete": true}
        - utter_filled_slots
        - utter_suggest_restaurant
    * affirm
        - utter_book_restaurant

Therefore it is useful being deliberate about which slots you featurize and which you don't. I.e. in this case, if the slots you want to fill are only relevant as arguments to an api-call, then it is advised to not featurize the slots and instead include an action which checks if all the slots are filled, such as ``DeactivateForm`` in ``form_actions.py`` and then store the result of this in a slot which will be featurized.

Follow up actions
^^^^^^^^^^^^^^^^^

There are cases which are less straightforward than asking a question of what someone wants and they tell you what they want in full text. For example, if we include another question: "Would you like vegetarian options at the restaurant?"
the user will not likely say "yes, vegetarian options", they are more likely to say "yes" or "no". For cases like this we use something known as a ``follow_up_action``. An example of how this is defined in the form object is:

.. code-block:: python
        fields = { ...,
                "vegetarian": {
                    "ask_utt": "utter_ask_vegetarian",
                    "clarify_utt": "utter_explain_vegetarian",
                    "follow_up_action": "vegetarian_parse"}
                }

In this form, whenever the ``ask_utt`` is performed, the next action after the user's message will always be the ``follow_up_action``, in this case ``"vegetarian_parse"``.

The code for this follow up action is then:

.. code-block:: python
    class VegetarianParse(Action):
        def name(self):
            return "vegetarian_parse"

        def run(self, dispatcher, tracker, domain, executor):
            latest_intent = tracker.latest_message['intent']['name']
            if latest_intent == 'affirm':
                return [SlotSet('vegetarian', True)]
            elif latest_intent == 'deny':
                return [SlotSet('vegetarian', False)]
            else:
                return []

This action picks up the most recent intent and sets a slot dependent on it. This will also allow the plan to move on to the next question.

How does it work?
^^^^^^^^^^^^^^^^^

It is worthwhile taking a brief look at the Form object to understand the workflow and how the different arguments interact with one another. The full object is in ``rasa_core.policies.forms``, but you can get an idea just from looking at the ``next_action_idx`` function:

.. code-block:: python

    def next_action(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> int

        out = self._run_through_queue(domain)
        if out is not None:
            # There are still actions in the queue, do the next one
            return out

        self.current_failures += 1
        if self.current_failures > self.max_turns:
            self.queue = [self.failure_action, self.finish_action]
            return self._run_through_queue(domain)

        intent = tracker.latest_message['intent']['name']
        self._update_requirements(tracker)

        if intent in self.exit_dict.keys():
            # actions in this dict should deactivate this form in the tracker
            self._exit_queue(intent, tracker)
            return self._run_through_queue(domain)

        elif intent in self.chitchat_dict.keys():
            self._chitchat_queue(intent, tracker)
            return self._run_through_queue(domain)

        elif self.details_intent and intent == self.details_intent:
            self._details_queue(intent, tracker)
            return self._run_through_queue(domain)

        still_to_ask = self.check_unfilled_slots(tracker)

        if len(still_to_ask) == 0:
            # if all the slots have been filled then queue finish actions
            self.queue = [self.finish_action]
            return self._run_through_queue(domain)
        else:
            # otherwise just ask to fill slots
            self.last_question = self._decide_next_question(still_to_ask, tracker)
            self.queue = self._question_queue(self.last_question)
            return self._run_through_queue(domain)

Forms work by queueing up a list of actions as soon as it is the bot's turn to speak again. There are several "queues" of actions that can be lined up. The most common one will be the ``_question_queue`` which contains the ``ask_utt`` for an unfilled slot and then listens (If there is a ``follow_up_acton`` the queue will have that action appended after the ``action_listen`` and will be the first action done before a new queue is made). Another queue is the finish queue, which will take the action listed as ``finish_action`` and execute it. The chitchat queue will, when presented with one of the keys of ``chitchat_dict``, perform the corresponding action and then repeat the question it previously asked. the details queue will perform the 'clarify_utt' action, say the previous question and then listen when being provided the ``details_intent``. The last queue is the exit dict which will, when presented with the intent key, perform the corresponding value action. The action itself must exit the Form by returning a ``StopForm`` event.

We intend forms to be used as a majority slot-filling exercise, which means that all intents are ignored except in the cases that:
- your ``follow_up_action`` explicitly deals with the intent
- any intent which is in ``[exit_dict.keys(), chitchat_dict.keys(), details_intent]`` is detected.


Example
-------

To see the forms in action or use them yourself, check out the formbot in ``rasa_core_sdk/examples/formbot``.