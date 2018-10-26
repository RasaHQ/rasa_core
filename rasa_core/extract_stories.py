from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.utils import EndpointConfig
import io
from tqdm import tqdm

CORE_HOST = "http://localhost:5005"
core_endpoint = EndpointConfig(CORE_HOST, token='coretoken')

def fetch_sender_ids():
    result = core_endpoint.request(method="get", subpath="/conversations")
    return result.json()

def fetch_story(sender_id):
    result = core_endpoint.request(
        method="get",
        subpath="/conversations/{}/story".format(sender_id))
    return result.text

def fetch_stories(n=None):
    sender_ids = fetch_sender_ids()
    if n:
        sender_ids = sender_ids[-n:]
    return [fetch_story(_id)
        for _id in sender_ids]

def evaluate(story):
    result = core_endpoint.request(
        subpath="/evaluate",
        content_type="text/markdown",
        data=story.encode('utf-8'))
    return result.json()

def is_successful(story):
    raise NotImplementedError()


if __name__ == "__main__":

    correct_predict = io.open("correct_predict.md", "w")
    incorrect_predict = io.open("incorrect_predict.md", "w")
    incorrect_memo = io.open("incorrect_memo.md", "w")
    N = 150

    all_stories = fetch_stories(n=N)
    for i, story in tqdm(enumerate(all_stories)):
        result = evaluate(story)
        if 'in_training_data_fraction' in result:
            memo_frac = result['in_training_data_fraction']
            if memo_frac < 1.:
                if is_successful(story):
                    correct_predict.write(story+"\n")
                else:
                    incorrect_predict.write(story+"\n")
            else:
                if not is_successful(story):
                    incorrect_memo.write(story+"\n")

    
