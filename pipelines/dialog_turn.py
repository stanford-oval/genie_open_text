from collections import OrderedDict
from typing import List


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        pipeline: str = None,
        engine: str = None,
    ):
        self.engine = engine
        self.pipeline = pipeline
        self.wall_time_seconds = (
            0  # how much time it took to generate this turn, in seconds
        )
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.dlg_history = None

        # retrieve_and_generate pipeline
        self.initial_search_query = None
        self.initial_search_query_time = None
        self.initial_search_results = []
        self.initial_search_result_titles = []
        self.initial_search_bullets = []

        # verify_and_correct pipeline
        self.gpt3_agent_utterance = None
        self.claims = []
        self.verification_retrieval_results = {}
        self.verification_result = {}

        # early_combine pipeline
        self.combined_evidences = []
        self.combined_utterance = None
        self.feedback = []
        self.feedback_scores = []
        self.refined_utterance = None

        # with_replacement pipeline
        self.unsupported_claims = []
        self.claim_purposes = []
        self.corrected_claims = []

        # for faithfulness evaluation
        self.claim_splitter = None

    def _summarize_vc_log(self):
        verification_summary = {}
        assert len(self.verification_result) == len(
            self.verification_retrieval_results
        ), "We need to have retrieved evidence for all claims"
        for key, value in self.verification_retrieval_results.items():
            claim_idx = int(key)
            v_ret_results = []
            for v in value:
                title, _, paragraph, _, score = tuple(v)
                v_ret_results.append(
                    {"title": title, "paragraph": paragraph, "score": round(score, 1)}
                )
            verification_summary[self.claims[claim_idx][0]] = OrderedDict(
                {
                    "label": self.verification_result[claim_idx]["label"],
                    "fixed_claim": self.verification_result[claim_idx]["fixed_claim"],
                    "retrieval_results": v_ret_results,
                }
            )
        return verification_summary

    def _summarize_rg_log(self):
        rg_summary = {
            "initial_search_query": self.initial_search_query,
            "initial_search_query_time": self.initial_search_query_time,
            "initial_search_bullets": self.initial_search_bullets,
            "initial_search_results": [],
        }

        for i in range(len(self.initial_search_results)):
            rg_summary["initial_search_results"].append(
                {
                    "title": self.initial_search_result_titles[i],
                    "paragraph": self.initial_search_results[i],
                    # 'bullets': self.initial_search_bullets,
                }
            )

        return rg_summary

    def _remove_claims_from_previous_turns(self, claims: List, object_dlg_history):
        """
        Copied from chatbot.py.

        Removes claims that are repeated from the last turn. This is often the result of LLM making a mistake while splitting claims.
        But even if it is actually a claim that the chatbot repeats, removing it here is beneficial as it will reduce repetitiveness.
        """
        previous_turn_claims = []
        for i in range(len(object_dlg_history)):
            previous_turn_claims.extend([c[0] for c in object_dlg_history[i].claims])
        claims = [c for c in claims if c[0] not in previous_turn_claims]

        return claims

    def log(self):
        """
        Returns a json object that assists human to evaluate the system faithfulness.
        """
        # Split the final output into claims for evaluation.
        claims = self.claim_splitter.split_claim(
            dialog_history=self.dlg_history,
            new_user_utterance=self.user_utterance,
            current_agent_utterance=self.agent_utterance,
            system_parameters={'engine': self.engine}
        )
        claims = self._remove_claims_from_previous_turns(claims, self.dlg_history)
        claims = [claim[0] for claim in claims]

        # combine fields into a more human-readable field
        verification_summary = self._summarize_vc_log()
        rg_summary = self._summarize_rg_log()

        return OrderedDict(
            {
                "engine": self.engine,
                "pipeline": self.pipeline,
                "wall_time_seconds": round(self.wall_time_seconds, 1),
                "user_utterance": self.user_utterance,
                "agent_utterance": self.agent_utterance,
                "split_claims_for_eval": claims,
                "initial_search_retrieval_result": rg_summary,
                "verification_retrieval_result": verification_summary,
            }
        )

    @staticmethod
    def utterance_list_to_dialog_history(utterance_list: List[str]):
        """
        The resulting dialog history will not have all the fields correctly initialized, since no information about e.g. search queries is available
        """
        dialog_history = []
        assert (
            len(utterance_list) % 2 == 1
        ), "The first turn is always the user, and the turn to be generated is always the agent, so the number of turns should be odd"
        for i in range(0, len(utterance_list) - 2, 2):
            dialog_history.append(
                DialogueTurn(
                    user_utterance=utterance_list[i],
                    agent_utterance=utterance_list[i + 1],
                )
            )
        user_utterance = utterance_list[-1]

        return dialog_history, user_utterance

    @staticmethod
    def dialog_history_to_utterance_list(dialog_history) -> List[str]:
        """
        Convert a list of DialogueTurns to a list of strings
        """
        utterance_list = []
        for turn in dialog_history:
            utterance_list.append(turn.user_utterance)
            utterance_list.append(turn.agent_utterance)
        return utterance_list
