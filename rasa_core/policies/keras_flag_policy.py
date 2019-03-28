import copy
import json
import logging
import os
import numpy as np
import tensorflow as tf
import warnings
from typing import Any, List, Dict, Text, Optional, Tuple

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer)
from rasa_core.featurizers import TrackerFeaturizer
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class KerasFlagPolicy(KerasPolicy):

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)
        shuffled_X, shuffled_y, shuffled_flags, _ = training_data.shuffled_X_y()
        print(shuffled_X.shape)
        print(shuffled_y.shape)
        exit()
        featurizer = self.featurizer.state_featurizer
        slot_start = featurizer.user_feature_len
        previous_start = slot_start + featurizer.slot_feature_len
        form_start = previous_start + featurizer.prev_act_feature_len

        shuffled_y = np.concatenate([shuffled_X[:, :, previous_start:form_start], shuffled_y[:, np.newaxis, :]], axis=1)[:, 1:, :]
        shuffled_forms = shuffled_X[:, :, form_start:]

        shuffled_flags = shuffled_X[:, :, flag_start:]
        shuffled_X = shuffled_X[:, :, :previous_start]

        shuffled_X = np.concatenate([shuffled_X, shuffled_y, shuffled_forms], axis=2)
        print(shuffled_X.shape)
        exit()
        # old X and FullDialogueFeaturizer
        # shuffled_X = np.concatenate([shuffled.X, shuffled.y], axis=2)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                if self.model is None:
                    self.model = self.model_architecture(
                        shuffled_X.shape[1:], shuffled_flags.shape[1:])

                logger.info("Fitting model with {} total samples and a "
                            "validation split of {}"
                            "".format(training_data.num_examples(),
                                      self.validation_split))
                # filter out kwargs that cannot be passed to fit
                params = self._get_valid_params(self.model.fit, **kwargs)

                self.model.fit(shuffled_X, shuffled_flags,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               **params)
                # the default parameter for epochs in keras fit is 1
                self.current_epoch = self.defaults.get("epochs", 1)
                logger.info("Done fitting keras policy model")

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        pass

    def predict_flag_probabilities(self,
                                   tracker: DialogueStateTracker,
                                   domain: Domain,
                                   event) -> List[float]:

        # noinspection PyPep8Naming
        X = self.featurizer.create_X([tracker], domain)
        y = np.array([[self.featurizer.state_featurizer.action_as_one_hot(event.action_name, domain)]])
        topics = np.array([[self.featurizer.state_featurizer.topic_as_one_hot(event.topic, domain)]])

        # new X and MaxHistoryFeaturizer
        featurizer = self.featurizer.state_featurizer
        slot_start = featurizer.user_feature_len
        previous_start = slot_start + featurizer.slot_feature_len
        form_start = previous_start + featurizer.prev_act_feature_len
        topic_start = form_start + featurizer.form_feature_len
        flag_start = topic_start + featurizer.topic_feature_len

        y = np.concatenate([X[:, :, previous_start:form_start], y], axis=1)[:, 1:, :]
        forms = X[:, :, form_start:topic_start]
        topics = np.concatenate([X[:, :, topic_start:flag_start], topics], axis=1)[:, 1:, :]
        flags = X[:, :, flag_start:]
        X = X[:, :, :previous_start]

        # X = np.concatenate([X, y, forms, topics, flags], axis=2)
        X = np.concatenate([X, y, forms, flags], axis=2)

        # old X and FullDialogueFeaturizer
        # data = self.featurize_for_training([tracker], domain)
        # if X.shape[1] > 1:
        #     y = np.concatenate([data.y, y], axis=1)
        #     topics = np.concatenate([data.topics, topics], axis=1)
        #
        # X = np.concatenate([X, y, topics], axis=2)

        with self.graph.as_default(), self.session.as_default():
            flags_pred = self.model.predict(X, batch_size=1)

        if len(flags_pred.shape) == 2:
            return flags_pred[-1].tolist()
        elif len(flags_pred.shape) == 3:
            return flags_pred[0, -1].tolist()
