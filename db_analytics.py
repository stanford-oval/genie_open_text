import argparse
import os
import pymongo
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

# set up the MongoDB connection
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
client = pymongo.MongoClient(CONNECTION_STRING)
db = client['wikichat']  # the database name is wikichat
dialog_db_collection = db['dialog_turns'] # the collection that stores dialog turns and their user ratings
dialog_db_collection.create_index("$**") # necessary to build an index before we can call sort()
preference_db_collection = db['preferences'] # the collection that stores information about what utterance users preferred
preference_db_collection.create_index("$**") # necessary to build an index before we can call sort()

rating_fields = ["user_naturalness_rating", "user_factuality_rating", "user_factuality_confidence"]

def get_experiment_rating_stats(dialog_db_collection, experiment_id):
    """
    Computes and returns aggregated rating statistics for each system in the given experiment.

    Parameters:
    -----------
    experiment_id : str
        id of the experiment to retrieve statistics for.

    Returns:
    --------
    A dictionary containing an item for each rating_field, e.g.
    'user_naturalness_rating': A dictionary of mean naturalness ratings for each system.
    """
    # Fetch all dialogs associated with the given experiment_id
    dialogs = dialog_db_collection.find({"experiment_id": experiment_id})
    # Convert the retrieved documents to a pandas DataFrame
    df = pd.DataFrame(dialogs)
    # Compute mean ratings for each system across all dialogs
    all_naturalness_ratings = [a[0] for a in df[["user_naturalness_rating"]].values.tolist()]
    print("all_naturalness_ratings = ", all_naturalness_ratings)
    
    aggregated_df = df[["system_name"] + rating_fields].groupby("system_name").mean(numeric_only=False)
    # Convert the DataFrame to a dictionary with the specified format
    return dict([(rating, aggregated_df[rating].to_dict()) for rating in rating_fields])

def get_experiment_preference_stats(preference_db_collection, experiment_id):
    """
    Computes and returns aggregated preferences statistics for each system in the given experiment.

    Parameters:
    -----------
    experiment_id : str
        id of the experiment to retrieve statistics for.

    Returns:
    --------
        A dictionary of win counts for each system.
    """
    # Fetch all preferences associated with the given experiment_id
    preferences = preference_db_collection.find({"experiment_id": experiment_id})
    # Convert the retrieved documents to a pandas DataFrame
    df = pd.DataFrame(preferences)
    # Unique id for each system pair
    df["system_pair"] = df.apply(
        lambda row: "_vs_".join(sorted([str(ls) for ls in [row["winner_system"]] + row["loser_system"]])),
        axis=1,
    )
    # Compute win counts for each system pair across all preferences
    win_counts = df.groupby(["system_pair", "winner_system"]) \
        .size() \
        .reset_index(name="win_count") \
        .groupby("system_pair") \
        .apply(lambda x: dict(zip(x["winner_system"], x["win_count"]))) \
        .to_dict()
    return win_counts

def format_experiment_outputs(dialog_db_collection, preference_db_collection, experiment_id, output_file, single_system, text_only):
    """
    Dump the contents of the given dialog and preference databases for a specific experiment to a text file.
    """
    with open(output_file.strip("txt") + "log", "w") as log_file, open(output_file, 'w') as output_file:

        # get all dialogs for the given experiment
        dialogs = dialog_db_collection.distinct("dialog_id", {"experiment_id": experiment_id})

        # loop over each dialog
        print("Found %d dialogs with this experiment id" % len(dialogs))
        for dialog_id in dialogs:
            print("dialog_id = ", dialog_id)

            # group turns by dialog_id
            turns = dialog_db_collection.find({"experiment_id": experiment_id, "dialog_id": dialog_id}).sort("turn_id")

            # group turns by (experiment_id, dialog_id, turn_id)
            turn_groups = {}
            for turn in turns:
                key = (experiment_id, dialog_id, turn['turn_id'])
                if key not in turn_groups:
                    turn_groups[key] = []
                turn_groups[key].append(turn)
            
            if not text_only:
                # print the experiment, dialog
                output_file.write(f"experiment_id={experiment_id}\n")
                output_file.write(f"dialog_id={dialog_id}\n")

            # loop over each turn group
            for _id, turn_group in turn_groups.items():
                turn_id = _id[2]

                # get the current agent utterances for each system
                current_user_utterance = turn_group[0]['user_utterance']
                output_file.write("User(human): " + current_user_utterance + "\n")
                agent_utterances = {}
                for i in range(len(turn_group)):
                    current_system = turn_group[i]['system_name']
                    current_agent_utterance = turn_group[i]['agent_utterance']
                    agent_utterances[current_system] = {
                        "agent_utterance": current_agent_utterance,
                        "ratings": dict([(rating, turn_group[i][rating]) for rating in rating_fields]),
                        "log_object": turn_group[i]['agent_log_object'],
                    }

                if single_system:
                    winner_system = current_system
                else:
                    # get the winner system for this turn
                    preference = preference_db_collection.find_one({"experiment_id": experiment_id, "dialog_id": dialog_id, "turn_id": turn_id})
                        
                    if not preference and not text_only:
                        output_file.write(f"preference not saved to db: {experiment_id, dialog_id, turn_id} \n")
                        json.dump(agent_utterances, output_file, indent=4)
                        output_file.write("\n")
                        continue
                    winner_system = preference['winner_system']
                selected_agent_utterance = agent_utterances[winner_system]["agent_utterance"]
                output_file.write("Chatbot(" + winner_system + "): " + selected_agent_utterance + "\n")
                agent_utterances["user_preference"] = winner_system
                
                if not text_only:
                    # write agent responses
                    json.dump(agent_utterances, output_file, indent=4)
                    output_file.write("\n")
                
                json.dump(agent_utterances[winner_system]["log_object"], log_file)
                log_file.write("\n")

            # add a separator between dialogs
            output_file.write("=====\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True, help='Where to write the outputs.')
    parser.add_argument('--experiment_id', type=str, required=True, help='The experiment_id')
    parser.add_argument('--single_system', action='store_true', help="True if the experiment only contains one system")
    parser.add_argument('--text_only', action='store_true', help="Only output the dialog text, and none of the logs or ratings")

    args = parser.parse_args()
    if not args.single_system:
        rating_stats = get_experiment_rating_stats(dialog_db_collection, args.experiment_id)
        print("rating_stats: ", rating_stats)
        preference_stats = get_experiment_preference_stats(preference_db_collection, args.experiment_id)
        print("preference_stats: ", preference_stats)
    format_experiment_outputs(dialog_db_collection, preference_db_collection, args.experiment_id, args.output_file, args.single_system, text_only=args.text_only)
