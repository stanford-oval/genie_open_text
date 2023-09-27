"""
The backend API that runs dialog agents, returns agent utterance to the front-end, and stores user data in a MongoDB database

The API has the following three functions that can be used by any front-end.
All inputs/outputs are string, except for `log_object` which is a json object and `turn_id` and `user_naturalness_rating` which are integers.
- `/chat`
Inputs: (experiment_id, new_user_utterance, dialog_id, turn_id, system_name)
Outputs: (agent_utterance, log_object)
Each time a user types something and clicks send, the front-end should make one call per system to /chat. So e.g. it should make two separate calls for two systems.

- `/user_rating`
Inputs: (experiment_id, dialog_id, turn_id, system_name, user_naturalness_rating, user_factuality_rating, user_factuality_confidence)
Outputs: None
When the user submits their ratings, the front-end should make one call per system to /user_rating. So e.g. it should make two separate calls for two systems.

- `/user_preference`
Inputs: (experiment_id, dialog_id, turn_id, winner_system, loser_systems)
Outputs: None
Each time the user selects one of the agent utterances over the other, you make one call to /user_preference.

`turn_id` starts from 0 and is incremented by 1 after a user and agent turn
"""

import time
from typing import List
from pipelines.dialog_turn import DialogueTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.chatbot import Chatbot
import argparse
import os
import json
import logging

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Api, reqparse
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pymongo


# set up the MongoDB connection
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
client = pymongo.MongoClient(CONNECTION_STRING)
db = client["wikichat"]  # the database name is wikichat
dialog_db_collection = db[
    "dialog_turns"
]  # the collection that stores dialog turns and their user ratings
dialog_db_collection.create_index(
    "$**"
)  # necessary to build an index before we can call sort()
preference_db_collection = db[
    "preferences"
]  # the collection that stores information about what utterance users preferred
preference_db_collection.create_index(
    "$**"
)  # necessary to build an index before we can call sort()
# The "schema" of dialog_db_collection is: {_id=(dialog_id, turn_id, system_name), experiment_id, dialog_id, turn_id, system_name, user_utterance, agent_utterance, agent_log_object, user_naturalness_rating}
# The "schema" of preference_db_collection is: {_id=(dialog_id, turn_id), experiment_id, dialog_id, turn_id, winner_system, loser_systems}

# set up the Flask app
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
CORS(app)
api = Api(app)

# default rate limit for all routs. Note that each turn may have multiple calls to each API below
# TODO add storage_uri to make rate limit tracking data persistent. Something like storage_uri="mongodb://localhost:27017"
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["500 per day"])

logging.getLogger("openai").setLevel(logging.ERROR)

logger = app.logger

arg_parser = argparse.ArgumentParser()
add_pipeline_arguments(arg_parser)

arg_parser.add_argument(
    "--ssl_certificate_file",
    type=str,
    default="/etc/letsencrypt/live/backend.wikichat.genie.stanford.edu/fullchain.pem",
    help="Where to read the SSL certificate for HTTPS",
)
arg_parser.add_argument(
    "--ssl_key_file",
    type=str,
    default="/etc/letsencrypt/live/backend.wikichat.genie.stanford.edu/privkey.pem",
    help="Where to read the SSL certificate for HTTPS",
)
arg_parser.add_argument(
    "--test",
    action="store_true",
    help="If set, no database query will be made, and mock outputs will be returned.",
)
arg_parser.add_argument(
    "--no_logging", action="store_true", help="Disables logging except for errors"
)

args = arg_parser.parse_args()
check_pipeline_arguments(args)

if args.no_logging:
    logging.basicConfig(
        level=logging.ERROR, format=" %(name)s : %(levelname)-8s : %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
    )


chatbot = Chatbot(args)

# The input arguments coming from the front-end
req_parser = reqparse.RequestParser()
req_parser.add_argument(
    "experiment_id",
    type=str,
    location="json",
    help="Identifier that differentiates data from different experiments.",
)
req_parser.add_argument(
    "dialog_id",
    type=str,
    location="json",
    help="Globally unique identifier for each dialog",
)
req_parser.add_argument(
    "turn_id", type=int, location="json", help="Turn number in the dialog"
)
req_parser.add_argument("user_naturalness_rating", type=int, location="json")
req_parser.add_argument("user_factuality_rating", type=int, location="json")
req_parser.add_argument("user_factuality_confidence", type=int, location="json")
req_parser.add_argument(
    "new_user_utterance", type=str, location="json", help="The new user utterance"
)
req_parser.add_argument(
    "system_name",
    type=str,
    location="json",
    help="The system to use for generating agent utterances",
)

# arguments for when a user makes a head-to-head comparison
req_parser.add_argument(
    "winner_system",
    type=str,
    location="json",
    help="The system that was preferred by the user in the current dialog turn",
)
req_parser.add_argument(
    "loser_systems",
    type=list,
    location="json",
    help="The system(s) that was not preferred by the user in the current dialog turn",
)


def extract_system_parameters(system: str):
    """
    Example input: generate[engine=text-davinci-003]
    """
    # return a dictionary of parameters inside []
    # if there are no parameters, return an empty dictionary
    if "[" not in system:
        return system, {}
    else:
        parameters = system[system.find("[") + 1 : system.find("]")]
        # split by comma
        parameters = parameters.split(",")
        # split by equal sign
        parameters = [p.split("=") for p in parameters]
        # convert to dictionary
        parameters = {p[0]: p[1] for p in parameters}
        # find system name
        system_name = system[: system.find("[")]
        return system_name, parameters


def filter_nons(request_args):
    return dict(
        [(k, request_args[k]) for k in request_args if request_args[k] is not None]
    )


def extract_db_history(dialog_id, turn_id) -> List[str]:
    db_dialog_history = []
    if turn_id > 0:
        # query DB to get previous turns
        # sort based on turn_id
        previous_turns = list(
            dialog_db_collection.find(
                {"dialog_id": dialog_id, "turn_id": {"$lt": turn_id}},
                {
                    "turn_id": 1,
                    "system_name": 1,
                    "user_utterance": 1,
                    "agent_utterance": 1,
                },
            ).sort("turn_id", pymongo.ASCENDING)
        )

        assert (len(previous_turns) == 0 and turn_id == 0) or len(
            previous_turns
        ) % turn_id == 0, "This turn should be the first turn of the dialog, or we should have a dialog turn document in the database per turn per system"

        # query db to see which system did the user prefer in the previous turns
        # sort based on turn_id
        previous_turn_preferences = list(
            preference_db_collection.find(
                {"dialog_id": dialog_id},
                {"turn_id": 1, "winner_system": 1, "loser_systems": 1},
            ).sort("turn_id", pymongo.ASCENDING)
        )

        for p in previous_turn_preferences:
            assert (
                p["turn_id"] < turn_id
            ), "We should not have any preferences for the current turn"

        if len(previous_turn_preferences) > 0:
            # we are in the middle of comparing two systems, so we need to figure out the actual history of the conversation
            assert (
                len(previous_turns) >= len(previous_turn_preferences)
                and len(previous_turns) % len(previous_turn_preferences) == 0
            ), "All previous turns need to have a user preference"
            # number_of_systems = len(previous_turns) // len(previous_turn_preferences)
            for t_id, turn_preference in enumerate(previous_turn_preferences):
                turn_preference["winner_system"]
                turn_with_winner_utterance = [
                    t
                    for t in previous_turns
                    if t["turn_id"] == t_id
                    and t["system_name"] == turn_preference["winner_system"]
                ]
                assert (
                    len(turn_with_winner_utterance) == 1
                ), "There should be only one utterance for each (turn_id, system_name)"
                db_dialog_history.append(
                    turn_with_winner_utterance[0]["user_utterance"]
                )
                db_dialog_history.append(
                    turn_with_winner_utterance[0]["agent_utterance"]
                )
        else:
            # there is only one system (we are not comparing multiple systems)
            for t in previous_turns:
                db_dialog_history.append(t["user_utterance"])
                db_dialog_history.append(t["agent_utterance"])

    return db_dialog_history


@app.route("/chat", methods=["POST"])
def chat():
    """
    Inputs: (experiment_id, new_user_utterance, dialog_id, turn_id, system_name)
    Outputs: (agent_utterance, log_object)
    """
    # logger.info('Entered /chat')
    request_args = req_parser.parse_args()
    logger.info("Input arguments received: %s", str(filter_nons(request_args)))

    experiment_id = request_args["experiment_id"]
    new_user_utterance = request_args["new_user_utterance"]
    dialog_id = request_args["dialog_id"]
    turn_id = request_args["turn_id"]
    system_name = request_args["system_name"]
    pipeline, system_parameters = extract_system_parameters(system_name)

    if args.test:
        logger.info("request from IP address %s", str(request.remote_addr))
        return {
            "agent_utterance": "Sample agent utterance. experiment_id=%s, new_user_utterance=%s, dialog_id=%s, turn_id=%d, system_name=%s"
            % (experiment_id, new_user_utterance, dialog_id, turn_id, system_name),
            "log_object": {"log object parameter 1": "log object value 1"},
        }

    # reconstruct the dialog history from DB
    try:
        db_dialog_history = extract_db_history(dialog_id, turn_id)
    except Exception as e:
        logger.exception(e)
        new_dlg_turn = DialogueTurn(
            user_utterance=new_user_utterance,
            agent_utterance="Database error while retrieving dialog history.",
            pipeline=pipeline,
        )
        turn_log = new_dlg_turn.log()
        return {"agent_utterance": new_dlg_turn.agent_utterance, "log_object": turn_log}

    # generate system response
    try:
        dialog_history, u = DialogueTurn.utterance_list_to_dialog_history(
            db_dialog_history + [new_user_utterance]
        )
        new_dlg_turn = chatbot.generate_next_turn(
            dialog_history, u, pipeline=pipeline, system_parameters=system_parameters
        )
        agent_utterance = new_dlg_turn.agent_utterance
        turn_log = new_dlg_turn.log()
        # logger.info('system output = %s', json.dumps(turn_log, indent=4))
    except Exception as e:
        logger.exception(e)
        new_dlg_turn = DialogueTurn(
            user_utterance=new_user_utterance,
            agent_utterance="Encountered an error while generating this response. Please refresh the page and try again.",
            pipeline=pipeline,
        )
        turn_log = new_dlg_turn.log()
        return {"agent_utterance": new_dlg_turn.agent_utterance, "log_object": turn_log}

    # save system response to DB
    try:
        logger.info(
            "Inserting new entry with _id = %s", str((dialog_id, turn_id, system_name))
        )
        dialog_db_collection.insert_one(
            {
                "_id": str((dialog_id, turn_id, system_name)),
                "experiment_id": experiment_id,
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "system_name": system_name,
                "user_utterance": new_user_utterance,
                "agent_utterance": agent_utterance,
                "agent_log_object": turn_log,
                "user_naturalness_rating": -1,
                "user_factuality_rating": False,
                "user_factuality_confidence": -1,
            }
        )
    except Exception as e:
        logger.exception(e)
        new_dlg_turn = DialogueTurn(
            user_utterance=new_user_utterance,
            agent_utterance="Database error while saving the current turn.",
            pipeline=pipeline,
        )
        turn_log = new_dlg_turn.log()
        return {"agent_utterance": new_dlg_turn.agent_utterance, "log_object": turn_log}

    # code for adding delay to faster pipielines, so that the user study subjects focus on quality not latency
    # time_spent = turn_log["wall_time_seconds"]
    # if time_spent < 15 and pipeline == "generate":
    # logger.info("Waiting for an extra %.2f seconds", 15 - time_spent)
    # time.sleep(15 - time_spent)

    return {"agent_utterance": agent_utterance, "log_object": turn_log}


@app.route("/user_rating", methods=["POST"])
def user_rating():
    """
    Inputs: (experiment_id, dialog_id, turn_id, system_name, user_naturalness_rating, user_factuality_rating, user_factuality_confidence)
    Outputs: None
    """
    # logger.info('Entered /user_rating')
    request_args = req_parser.parse_args()
    logger.info("Input arguments received: %s", str(filter_nons(request_args)))

    experiment_id = request_args["experiment_id"]
    dialog_id = request_args["dialog_id"]
    turn_id = request_args["turn_id"]
    system_name = request_args["system_name"]
    user_naturalness_rating = request_args["user_naturalness_rating"]
    user_factuality_rating = request_args["user_factuality_rating"]
    user_factuality_confidence = request_args["user_factuality_confidence"]

    if args.test:
        return {}

    matching_turn = list(
        dialog_db_collection.find(
            {"dialog_id": dialog_id, "turn_id": turn_id, "system_name": system_name},
            {
                "experiment_id": 1,
                "user_naturalness_rating": 1,
                "user_factuality_rating": 1,
                "user_factuality_confidence": 1,
            },
        )
    )

    assert (
        len(matching_turn) == 1
    ), "There should be exactly one (dialog_id, turn_id, system_name)"
    matching_turn = matching_turn[0]

    assert (
        matching_turn["user_naturalness_rating"] == -1
    ), "This turn has already been rated by the user."
    assert (
        matching_turn["user_factuality_rating"] == False
    ), "This turn has already been rated by the user."
    assert (
        matching_turn["user_factuality_confidence"] == -1
    ), "This turn has already been rated by the user."
    assert (
        matching_turn["experiment_id"] == experiment_id
    ), "experiment_id of the found document does not match the experiment_id received from the front-end"  # we check this just to be safe

    # update the document
    dialog_db_collection.update_one(
        {"dialog_id": dialog_id, "turn_id": turn_id, "system_name": system_name},
        {
            "$set": {
                "user_naturalness_rating": user_naturalness_rating,
                "user_factuality_rating": user_factuality_rating,
                "user_factuality_confidence": user_factuality_confidence,
            }
        },
    )

    return {}


@app.route("/user_preference", methods=["POST"])
def user_preference():
    """
    Inputs: (experiment_id, dialog_id, turn_id, winner_system, loser_systems)
    Outputs: None
    """
    # logger.info('Entered /user_preference')
    request_args = req_parser.parse_args()
    logger.info("Input arguments received: %s", str(filter_nons(request_args)))

    experiment_id = request_args["experiment_id"]
    dialog_id = request_args["dialog_id"]
    turn_id = request_args["turn_id"]
    winner_system = request_args["winner_system"]
    loser_systems = request_args["loser_systems"]

    if args.test:
        return {}

    # check these just to be safe
    matching_turn = list(
        dialog_db_collection.find(
            {"dialog_id": dialog_id, "turn_id": turn_id, "system_name": winner_system}
        )
    )
    assert (
        len(matching_turn) == 1
    ), "There should be exactly one (dialog_id, turn_id, system_name)"
    assert (
        matching_turn[0]["experiment_id"] == experiment_id
    ), "experiment_id of the found document does not match the experiment_id received from the front-end"  # we check this just to be safe
    for loser_system in loser_systems:
        matching_turn = list(
            dialog_db_collection.find(
                {
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
                    "system_name": loser_system,
                }
            )
        )
        assert len(matching_turn) == 1, (
            "There should be exactly one (dialog_id, turn_id, system_name) but found %s"
            % str(matching_turn)
        )
        assert (
            matching_turn[0]["experiment_id"] == experiment_id
        ), "experiment_id of the found document does not match the experiment_id received from the front-end"  # we check this just to be safe

    logger.info("Inserting new entry with _id = %s", str((dialog_id, turn_id)))
    preference_db_collection.insert_one(
        {
            "_id": str((dialog_id, turn_id)),
            "experiment_id": experiment_id,
            "dialog_id": dialog_id,
            "turn_id": turn_id,
            "winner_system": winner_system,
            "loser_system": loser_systems,
        }
    )

    return {}


if __name__ == "__main__":
    context = (args.ssl_certificate_file, args.ssl_key_file)
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=args.test,
        use_reloader=False,
        ssl_context=context,
    )


# Example curl command for testing:
# curl https://127.0.0.1:5001/chat -d '{"experiment_id": "test_experiment", "dialog_id": "test_dialog", "turn_id": 0, "system_name": "retrieve_and_generate", "new_user_utterance": "who stars in the wandering earth 2?"}' -X POST -H 'Content-Type: application/json'
