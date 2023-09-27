import unittest
from pymongo import MongoClient
import os

# Import the functions to test
from db_analytics import get_experiment_rating_stats, get_experiment_preference_stats


# Define the test case class
class TestExperimentStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Connect to the database and create some sample data
        cls.CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
        cls.client = MongoClient(cls.CONNECTION_STRING)
        cls.db = cls.client["wikichat_test_experiment_stats"]
        cls.dialog_db_collection = cls.db[
            "dialog_turns"
        ]  # the collection that stores dialog turns and their user ratings
        cls.dialog_db_collection.create_index(
            "$**"
        )  # necessary to build an index before we can call sort()
        cls.preference_db_collection = cls.db[
            "preferences"
        ]  # the collection that stores information about what utterance users preferred
        cls.preference_db_collection.create_index(
            "$**"
        )  # necessary to build an index before we can call sort()
        cls.dialog_db_collection.insert_many(
            [
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_1",
                    "user_naturalness_rating": 0.8,
                    "user_factuality_rating": 0.9,
                    "user_factuality_confidence": 1,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_1",
                    "user_naturalness_rating": 0.9,
                    "user_factuality_rating": 0.7,
                    "user_factuality_confidence": 0,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_1",
                    "user_naturalness_rating": 0.7,
                    "user_factuality_rating": 0.4,
                    "user_factuality_confidence": 1,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_2",
                    "user_naturalness_rating": 0.6,
                    "user_factuality_rating": 0.5,
                    "user_factuality_confidence": 1,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_2",
                    "user_naturalness_rating": 0.5,
                    "user_factuality_rating": 0.6,
                    "user_factuality_confidence": 1,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_2",
                    "user_naturalness_rating": 0.7,
                    "user_factuality_rating": 0.8,
                    "user_factuality_confidence": 1,
                },
                {
                    "experiment_id": "test_experiment",
                    "system_name": "sys_3",
                    "user_naturalness_rating": 0.1,
                    "user_factuality_rating": 0.1,
                    "user_factuality_confidence": 0,
                },
            ]
        )
        cls.preference_db_collection.insert_many(
            [
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_1",
                    "loser_system": ["sys_2"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_1",
                    "loser_system": ["sys_2"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_2",
                    "loser_system": ["sys_1"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_3",
                    "loser_system": ["sys_2"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_3",
                    "loser_system": ["sys_4"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_3",
                    "loser_system": ["sys_4"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_3",
                    "loser_system": ["sys_4"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_4",
                    "loser_system": ["sys_3"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_5",
                    "loser_system": ["sys_6", "sys_7"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_6",
                    "loser_system": ["sys_5", "sys_7"],
                },
                {
                    "experiment_id": "test_experiment",
                    "winner_system": "sys_6",
                    "loser_system": ["sys_5", "sys_7"],
                },
            ]
        )

    @classmethod
    def tearDownClass(cls):
        # Remove the sample data from the database
        cls.dialog_db_collection.delete_many({})
        cls.preference_db_collection.delete_many({})
        cls.client.drop_database("wikichat_test_experiment_stats")

    def test_get_experiment_rating_stats(self):
        # Test the get_experiment_stats function with sample data
        expected_output = {
            "user_naturalness_rating": {"sys_1": 0.79999, "sys_2": 0.6, "sys_3": 0.1},
            "user_factuality_rating": {
                "sys_1": 0.66666,
                "sys_2": 0.63333,
                "sys_3": 0.1,
            },
            "user_factuality_confidence": {"sys_1": 0.66666, "sys_2": 1, "sys_3": 0},
        }
        actual_output = get_experiment_rating_stats(
            self.dialog_db_collection, "test_experiment"
        )
        for metric, systems in expected_output.items():
            for system, rating in systems.items():
                self.assertAlmostEqual(rating, actual_output[metric][system], places=3)

    def test_get_experiment_preference_stats(self):
        # Test the get_experiment_preference_stats function with sample data
        expected_output = {
            "sys_1_vs_sys_2": {"sys_1": 2, "sys_2": 1},
            "sys_2_vs_sys_3": {"sys_3": 1},
            "sys_3_vs_sys_4": {"sys_3": 3, "sys_4": 1},
            "sys_5_vs_sys_6_vs_sys_7": {"sys_5": 1, "sys_6": 2},
        }
        actual_output = get_experiment_preference_stats(
            self.preference_db_collection, "test_experiment"
        )
        for system_pair, system_count in expected_output.items():
            for system, count in system_count.items():
                self.assertEqual(count, actual_output[system_pair][system])


# Run the tests
if __name__ == "__main__":
    unittest.main()
