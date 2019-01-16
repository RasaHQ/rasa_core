import copy
import json
import logging
import os
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


class KerasTopicPolicy(KerasPolicy):

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)
        shuffled = training_data.shuffled()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                if self.model is None:
                    self.model = self.model_architecture(
                        shuffled.X.shape[1:], shuffled.topics.shape[1:])

                logger.info("Fitting model with {} total samples and a "
                            "validation split of {}"
                            "".format(training_data.num_examples(),
                                      self.validation_split))
                # filter out kwargs that cannot be passed to fit
                params = self._get_valid_params(self.model.fit, **kwargs)

                self.model.fit(shuffled.X, shuffled.topics,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               **params)
                # the default parameter for epochs in keras fit is 1
                self.current_epoch = self.defaults.get("epochs", 1)
                logger.info("Done fitting keras policy model")

    def continue_training(self,
                          training_trackers: List[DialogueStateTracker],
                          domain: Domain,
                          **kwargs: Any) -> None:
        """Continues training an already trained policy."""

        # takes the new example labelled and learns it
        # via taking `epochs` samples of n_batch-1 parts of the training data,
        # inserting our new example and learning them. this means that we can
        # ask the network to fit the example without overemphasising
        # its importance (and therefore throwing off the biases)

        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 50)

        with self.graph.as_default(), self.session.as_default():
            for _ in range(epochs):
                training_data = self._training_data_for_continue_training(
                    batch_size, training_trackers, domain)

                # fit to one extra example using updated trackers
                self.model.fit(training_data.X, training_data.topics,
                               epochs=self.current_epoch + 1,
                               batch_size=len(training_data.topics),
                               verbose=0,
                               initial_epoch=self.current_epoch)

                self.current_epoch += 1

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        return None

    def predict_topic_probabilities(self,
                                    tracker: DialogueStateTracker,
                                    domain: Domain) -> List[float]:
        # noinspection PyPep8Naming
        X = self.featurizer.create_X([tracker], domain)

        with self.graph.as_default(), self.session.as_default():
            topics_pred = self.model.predict(X, batch_size=1)

        if len(topics_pred.shape) == 2:
            return topics_pred[-1].tolist()
        elif len(topics_pred.shape) == 3:
            return topics_pred[0, -1].tolist()
