from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm
import argparse
import logging
import io
import os

from rasa_core import utils

logger = logging.getLogger(__name__)

# constants
UNKNOWN = "UNKNOWN"
SUCCESSFUL = "SUCCESSFUL"
UNSUCCESSFUL = "UNSUCCESSFUL"
MEMORIZED = "MEMORIZED"
PREDICTED = "PREDICTED"

template = {
    SUCCESSFUL: {
        MEMORIZED: [],
        PREDICTED: [],
    },
    UNSUCCESSFUL: {
        MEMORIZED: [],
        PREDICTED: [],
    },
}

def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description='fetches conversations from a rasa core server to use for training')
    parser = add_args_to_parser(parser)

    utils.add_logging_option_arguments(parser)
    return parser

def add_args_to_parser(parser):

    def try_parse_date(date):
        try:
            return datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            message = "date had an invalid format '{}' cannot be parsed.".format(s)
            raise argparse.ArgumentTypeError(message)

    parser.add_argument(
            '-o', '--out',
            type=str,
            required=False,
            help="directory to persist the trained model in",
            default="extracted_stories"
    )
    parser.add_argument(
            '--host',
            type=str,
            required=True,
            help="hostname where rasa core server is running"
    )
    parser.add_argument(
            '-t', '--token',
            type=str,
            required=False,
            help="rasa core server token"
    )
    parser.add_argument(
            "-d", "--date", 
            help="date in format YYYY-MM-DD , only fetch stories with a latest_message after this date",
            required=False, 
            type=try_parse_date)
    parser.add_argument(
            "-N", "--num_stories", 
            help="maximum number of stories to evaluate, defaults to 100",
            required=False,
            default=100,
            type=int)
    return parser

def fetch_sender_ids(endpoint):
    result = endpoint.request(method="get", subpath="/conversations")
    return result.json()

def fetch_story(endpoint, sender_id):
    result = endpoint.request(
        method="get",
        subpath="/conversations/{}/story".format(sender_id))
    return result.text

def fetch_stories(endpoint, max_stories=None):
    sender_ids = fetch_sender_ids(endpoint)
    if max_stories:
        sender_ids = sender_ids[-max_stories:]
    return [fetch_story(endpoint, _id)
        for _id in sender_ids]

def evaluate(story):
    memorized = MEMORIZED
    response = endpoint.request(
        subpath="/evaluate",
        content_type="text/markdown",
        data=story.encode('utf-8'))
    result = response.json()
    if result['in_training_data_fraction'] < 1:
        memorized = PREDICTED
    return result, memorized

def write_results(results, output_dir=None):

    def fpath(goal, success, memo):
        name = "{}_{}_{}.md".format(goal, success, memo)
        if output_dir:
            name = os.path.join(output_dir, name)
        return name

    for goal in results.keys():
        for success in results[goal].keys():
            for memo in results[goal][success].keys():
                stories = results[goal][success][memo]
                if not stories:
                    continue

                file_path = fpath(goal, success, memo)
                utils.create_dir_for_file(file_path)
                with io.open(file_path, "w") as f:
                    f.write("\n".join(stories))

def goal_success(story):
    user_goal, success = UNKNOWN, UNSUCCESSFUL

    if "signup_newsletter" in story:
        user_goal = "signup_newsletter"
        if "action_subscribe_newsletter" in story:
            success = SUCCESSFUL

    return user_goal, success
    

if __name__ == "__main__":

    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    endpoint = utils.EndpointConfig(
        cmdline_args.host,
        token=cmdline_args.token)

    goals_to_ignore = []
    results = {}

    all_stories = fetch_stories(endpoint, max_stories=cmdline_args.num_stories)
 
    for story in tqdm(all_stories):

        goal, success = goal_success(story)
        if goal in goals_to_ignore:
            continue
        _, memorized = evaluate(story)
        if goal not in results:
            results[goal] = template.copy()

        results[goal][success][memorized].append(story)

    write_results(results, output_dir=cmdline_args.out)
