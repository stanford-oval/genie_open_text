"""
Verify and correct single-turn responses from GPT-3
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List
import re
import requests
import logging
import numpy as np

from .dialog_turn import DialogueTurn
from .llm import llm_generate
from .split_claim import ClaimSplitter
from .utils import is_everything_verified, extract_year, extract_vote

logger = logging.getLogger(__name__)


class Chatbot:
    """
    A stateless chatbot. Stateless means that it does not store the history of the dialog in itself, but requires it as an input
    """

    def __init__(self, args) -> None:
        self.args = args
        if args.pipeline == "verify_and_correct" or args.pipeline == "early_combine" \
                or args.pipeline == "genie":
            self.claim_splitter = ClaimSplitter(args)
            self.evi_num = args.evi_num
        self.colbert_endpoint = args.colbert_endpoint

    def generate_next_turn(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            pipeline: str,
            system_parameters: dict = {},
            original_reply: str = "",
    ):
        """
        Generate the next turn of the dialog
        system_parameters: can override self.args. Only supports "engine" for now
        """
        # throw error if system_parameters contains keys that are not supported
        for key in system_parameters:
            assert key in ["engine"], f"Unsupported system_parameter key: {key}"

        start_time = time.time()

        if pipeline == "generate":  # Baseline.
            reply = self._generate_only(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                system_parameters=system_parameters,
            )
            new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
            new_dlg_turn.gpt3_agent_utterance = reply
            new_dlg_turn.agent_utterance = reply
        elif pipeline == "genie":
            new_dlg_turn = self.early_combine_with_replacement_pipeline(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                system_parameters=system_parameters,
                original_reply=original_reply,
            )
        else:
            raise ValueError

        new_dlg_turn.engine = system_parameters.get("engine", self.args.engine)
        new_dlg_turn.pipeline = pipeline

        end_time = time.time()
        new_dlg_turn.wall_time_seconds = end_time - start_time
        new_dlg_turn.dlg_history = object_dlg_history
        new_dlg_turn.claim_splitter = self.claim_splitter

        return new_dlg_turn

    def early_combine_with_replacement_pipeline(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            system_parameters: dict,
            original_reply: str = "",
    ):
        new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
        if not original_reply:
            original_reply = self._generate_only(
                object_dlg_history,
                new_user_utterance,
                system_parameters=system_parameters,
            )
        new_dlg_turn.gpt3_agent_utterance = original_reply

        # gather evidence from two routs in parallel
        with ThreadPoolExecutor(2) as executor:
            search_summary = executor.submit(
                self._search_and_summarize,
                object_dlg_history,
                new_user_utterance,
                new_dlg_turn,
                system_parameters=system_parameters,
            )
            supported_claims = executor.submit(
                self._split_and_fact_check,
                object_dlg_history,
                new_user_utterance,
                original_reply,
                new_dlg_turn,
                system_parameters=system_parameters,
                with_correct=True
            )
        search_summary = search_summary.result()
        supported_claims = supported_claims.result()

        combined_evi = supported_claims + search_summary
        new_dlg_turn.combined_evidences = combined_evi

        if not combined_evi:
            logger.info("Combined evidence is empty")
            if new_dlg_turn.initial_search_query is None:
                # No search needed, so return the original chitchat response
                new_dlg_turn.combined_utterance = original_reply
            else:
                # will become more conversational after refinement
                new_dlg_turn.combined_utterance = ("Sorry, I cannot find information on the website. "
                                                   "You can raise a new question in our online community.")

        logger.info("Combining the information.")
        new_dlg_turn.combined_utterance = self._reply_using_combined_evidence(
            original_reply,
            object_dlg_history,
            new_user_utterance,
            combined_evi,
            system_parameters=system_parameters,
        )
        new_dlg_turn.agent_utterance = new_dlg_turn.combined_utterance

        return new_dlg_turn

    def _handle_search_prompt_output_wikichat(
            self,
            search_prompt_output: str,
            new_dlg_turn: DialogueTurn,
            num_paragraphs,
            summarize_results: bool,
            system_parameters: dict,
    ):
        """
        LEGACY: This is the old version used by wikichat.

        Updates `new_dlg_turn` with logs
        A sample output is: Yes. You Google "James E. Webb the administrator of NASA". The year of the results is "not important".]
        """

        search_prompt_output = search_prompt_output.strip()
        search_pattern = r'Yes\. You.*"([^"]*)".* The year of the results is "([^=]*)"\.]?'
        search_match = re.match(search_pattern, search_prompt_output)

        if search_prompt_output.startswith("No"):
            pass
        elif search_match:
            search_query = search_match.group(1)
            search_query_time = search_match.group(2)
            y = extract_year(title="", passage=search_query)
            if len(y) > 0:
                logger.info("Overriding query year")
                search_query_time = y[0]
            logger.info("search_query = %s", search_query)
            logger.info("search_query_time = %s", search_query_time)

            if self.args.reranking_method == "none":
                paragraphs, scores, titles = self._colbert_retrieve(
                    query=search_query, num_paragraphs=num_paragraphs
                )
            else:
                # retrieve more so that we can match the dates
                paragraphs, scores, titles = self._colbert_retrieve(
                    query=search_query,
                    num_paragraphs=num_paragraphs,
                    rerank=search_query_time,
                    num_paragraphs_for_reranking=10,
                )

            if summarize_results:
                # summarize the best search results
                summaries = llm_generate(
                    template_file=self.args.summary_prompt,
                    prompt_parameter_values=[
                        {"title": t, "article": p, "query": search_query}
                        for (t, p) in zip(titles, paragraphs)
                    ],
                    engine=system_parameters.get("engine", self.args.engine),
                    max_tokens=200,
                    temperature=0.0,
                    top_p=0.5,
                    stop_tokens=None,
                    postprocess=False,
                    ban_line_break_start=False,
                )
                bullets = []
                for s in summaries:
                    if s.startswith("Yes. "):
                        # necessary for distilled models
                        s = s[5:]
                    if s.startswith("None"):
                        continue
                    for b in s.split("\n-"):
                        b = b.strip()
                        if len(b) == 0:
                            continue
                        if not b.endswith("."):
                            # most likely a partial generation that was cut off because of max_tokens
                            continue
                        bullets.append(b.strip("- "))

                if len(bullets) == 0:
                    bullets = []
            else:
                bullets = None

            # log everything
            new_dlg_turn.initial_search_query = search_query
            new_dlg_turn.initial_search_query_time = search_query_time
            new_dlg_turn.initial_search_results = paragraphs
            new_dlg_turn.initial_search_result_titles = titles
            new_dlg_turn.initial_search_bullets = bullets
        else:
            raise ValueError(
                "Search prompt's output is invalid: %s" % search_prompt_output
            )

    def _handle_search_prompt_output(
            self,
            search_prompt_output: str,
            new_dlg_turn: DialogueTurn,
            num_paragraphs,
            summarize_results: bool,
            system_parameters: dict,
    ):
        """
        Updates `new_dlg_turn` with logs
        A sample output is: Yes. You search "James E. Webb the administrator of NASA".]

        The retrieved documents are reranked based on the number of voting.
        """

        search_prompt_output = search_prompt_output.strip()
        search_pattern = r'Yes\. You.*"([^"]*)".*\.]?'
        search_match = re.match(search_pattern, search_prompt_output)

        if search_prompt_output.startswith("No"):
            pass
        elif search_match:
            search_query = search_match.group(1)

            if self.args.reranking_method == "none":
                paragraphs, scores, titles = self._colbert_retrieve(
                    query=search_query, num_paragraphs=num_paragraphs
                )
            else:
                # retrieve more so that we can match the dates
                paragraphs, scores, titles = self._colbert_retrieve(
                    query=search_query,
                    num_paragraphs=num_paragraphs,
                    rerank="voting",
                    num_paragraphs_for_reranking=10,
                )

            if summarize_results:
                # summarize the best search results
                summaries = llm_generate(
                    template_file=self.args.summary_prompt,
                    prompt_parameter_values=[
                        {"title": t, "article": p, "query": search_query}
                        for (t, p) in zip(titles, paragraphs)
                    ],
                    engine=system_parameters.get("engine", self.args.engine),
                    max_tokens=200,
                    temperature=0.0,
                    top_p=0.5,
                    stop_tokens=None,
                    postprocess=False,
                    ban_line_break_start=False,
                )
                bullets = []
                for s in summaries:
                    if s.startswith("Yes. "):
                        # necessary for distilled models
                        s = s[5:]
                    if s.startswith("None"):
                        continue
                    for b in s.split("\n-"):
                        b = b.strip()
                        if len(b) == 0:
                            continue
                        if not b.endswith("."):
                            # most likely a partial generation that was cut off because of max_tokens
                            continue
                        bullets.append(b.strip("- "))

                if len(bullets) == 0:
                    bullets = []
            else:
                bullets = None

            # log everything
            new_dlg_turn.initial_search_query = search_query
            new_dlg_turn.initial_search_results = paragraphs
            new_dlg_turn.initial_search_result_titles = titles
            new_dlg_turn.initial_search_bullets = bullets
        else:
            raise ValueError(
                "Search prompt's output is invalid: %s" % search_prompt_output
            )

    def _search_and_summarize(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            new_dlg_turn: DialogueTurn,
            system_parameters: dict,
    ):
        logger.info("Running initial search.")
        search_prompt_output = llm_generate(
            template_file=self.args.initial_search_prompt,
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "new_user_utterance": new_user_utterance,
                "force_search": False,
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=50,
            temperature=0.0,
            top_p=0.5,
            stop_tokens=["\n"],
            postprocess=False,
            ban_line_break_start=True,
        )
        logger.info("Summarizing search results.")
        self._handle_search_prompt_output(
            search_prompt_output=search_prompt_output,
            new_dlg_turn=new_dlg_turn,
            num_paragraphs=3,
            summarize_results=True,
            system_parameters=system_parameters,
        )
        return new_dlg_turn.initial_search_bullets

    def _split_and_fact_check(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            original_reply: str,
            new_dlg_turn: DialogueTurn,
            system_parameters: dict,
            with_correct: bool = False,
            replace_unsupported_claim: bool = False
    ):
        logger.info("Splitting the claims.")
        claims = self.claim_splitter.split_claim(
            dialog_history=object_dlg_history,
            new_user_utterance=new_user_utterance,
            current_agent_utterance=original_reply,
            system_parameters=system_parameters,
        )
        claims = self._remove_claims_from_previous_turns(claims, object_dlg_history)

        new_dlg_turn.claims = claims
        if not claims:
            return []

        # retrieve evidence
        ret_output = self._retrieve_evidences(claims)

        # verify claims
        logger.info("Verifying claims.")
        ver_output = self._verify_claims(
            claims,
            ret_output,
            object_dlg_history,
            new_user_utterance,
            original_reply,
            system_parameters=system_parameters,
            with_correct=with_correct
        )

        new_dlg_turn.verification_retrieval_results = ret_output
        new_dlg_turn.verification_result = ver_output

        supported_claims = []
        for claim_id, label_fix in enumerate(ver_output):
            verification_label, fixed_claim = (
                label_fix["label"],
                label_fix["fixed_claim"],
            )
            if verification_label == "SUPPORTS":
                supported_claims.append(fixed_claim)
            else:
                if with_correct and fixed_claim != "":
                    supported_claims.append(fixed_claim)
                elif replace_unsupported_claim:
                    evidences = ret_output[claim_id][: self.evi_num]
                    claim_purpose, corrected_claim = self._replace(
                        object_dlg_history,
                        new_user_utterance,
                        original_reply,
                        claims[claim_id][0],  # claims[claim_id]: (claim, year)
                        evidences,
                        system_parameters=system_parameters
                    )
                    new_dlg_turn.unsupported_claims.append(claims[claim_id][0])
                    new_dlg_turn.claim_purposes.append(claim_purpose)
                    new_dlg_turn.corrected_claims.append(corrected_claim)
                    if "NO IDEA" not in corrected_claim:
                        supported_claims.append(corrected_claim)

        return supported_claims

    def _verify_and_correct_reply(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            original_reply: str,
            new_dlg_turn: DialogueTurn,
            system_parameters: dict,
    ) -> str:
        """
        Verifies and corrects `original_reply` given the dialog history
        Updates `new_dlg_turn` with logs
        Returns corrected reply
        """
        # split claims
        # the returned "claims" is a list of tuples (claim, year)
        claims = self.claim_splitter.split_claim(
            dialog_history=object_dlg_history,
            new_user_utterance=new_user_utterance,
            current_agent_utterance=original_reply,
            system_parameters=system_parameters,
        )
        claims = self._remove_claims_from_previous_turns(claims, object_dlg_history)
        if not claims:
            return original_reply
        new_dlg_turn.claims = claims

        # retrieve evidence
        ret_output = self._retrieve_evidences(claims)

        # verify claims
        ver_output = self._verify_claims(
            claims,
            ret_output,
            object_dlg_history,
            new_user_utterance,
            original_reply,
            system_parameters=system_parameters,
        )

        # update dialog turn
        new_dlg_turn.verification_retrieval_results = ret_output
        new_dlg_turn.verification_result = ver_output
        if is_everything_verified(ver_output):
            return original_reply

        # correction
        corrected_reply = original_reply
        fixed_claims = []
        for label_fix in ver_output:
            verification_label, fixed_claim = (
                label_fix["label"],
                label_fix["fixed_claim"],
            )
            if (
                    verification_label == "SUPPORTS"
            ):  # if the claim is already correct, no need to fix
                continue
            fixed_claims.append(fixed_claim)

        assert len(fixed_claims) > 0
        corrected_reply = self._correct(
            original_reply,
            object_dlg_history,
            new_user_utterance,
            fixed_claims,  # corrected claim for REFUTE and "I'm not sure" for NOT ENOUGH INFO claims.
            system_parameters=system_parameters,
        )

        return corrected_reply

    def _generate_only(
            self,
            dialog_history: List[DialogueTurn],
            new_user_utterance: str,
            system_parameters: dict,
    ) -> str:
        """
        Generate baseline GPT3 response
        Args:
            - `dialog_history` (list): previous turns
        Returns:
            - `reply`(str): GPT3 original response
        """
        reply = llm_generate(
            template_file="prompts/baseline_chatbot.prompt",
            prompt_parameter_values={
                "dlg": dialog_history,
                "new_user_utterance": new_user_utterance,
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            stop_tokens=["\n"],
            top_p=self.args.top_p,
            frequency_penalty=self.args.frequency_penalty,
            presence_penalty=self.args.presence_penalty,
            postprocess=True,
            ban_line_break_start=True,
        )

        return reply

    def _correct(
            self,
            original_reply,
            object_dlg_history,
            last_user_utterance,
            fixed_claims,
            system_parameters: dict,
    ):
        """
        Given context + original response + evidence for a claim, prompt GPT3 to fix the original response

        Args:
            - `original_reply`(str): GPT3 original response
            - `object_dlg_history`(list): list of previous DialogueTurns
            - `last_user_utterance` (str): last user utterance
            - `fixed_claims` (list): list of fixed claims
        Returns:
            - `corrected_reply`(str): corrected GPT3 response
        """
        # correction prompt's context should be in one line
        correction_reply = llm_generate(
            template_file="prompts/correction_combiner.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "last_user_utterance": last_user_utterance,
                "original_reply": original_reply,
                "fixed_claims": fixed_claims,
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=self.args.max_tokens,
            temperature=0,
            stop_tokens=["\n"],
            top_p=self.args.top_p,
            frequency_penalty=self.args.frequency_penalty,
            presence_penalty=self.args.presence_penalty,
            postprocess=True,
            ban_line_break_start=True,
        )

        return correction_reply

    def _replace(
            self,
            object_dlg_history: List[DialogueTurn],
            new_user_utterance: str,
            original_reply: str,
            unsupported_claim: str,
            evidences: list,
            system_parameters: dict
    ):
        """
        Given context + original response + evidence for an unsupported claim, prompt LLM to mine related information
        to replace the old claim.

        Args:
            - `object_dlg_history`(list): list of previous DialogueTurns
            - `new_user_utterance` (str): last user utterance
            - `original_reply`(str): LLM original response
            - `unsupported_claim`(str): "NOT ENOUGH INFO" or "REFUTED" claim
            - `evidences`(list): list of evidences
        Returns:
            - `claim_purpose`(str): question that the original claim tries to answer
            - `corrected_claim`(str): corrected claim, empty string if the claim cannot be corrected
        """
        initiative_and_revised_claim = llm_generate(
            template_file="prompts/replace_claim.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "new_user_utterance": new_user_utterance,
                "original_reply": original_reply,
                "evidence_titles": [e[0] for e in evidences],
                "evidence_texts": [e[2] for e in evidences],
                "claim": unsupported_claim
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=100,
            temperature=0,
            stop_tokens=None,
            postprocess=False
        )
        initiative_and_revised_claim = initiative_and_revised_claim.strip()
        pattern = r'"(.*)"\nThe supported answer to this is: "(.*)"'
        match_result = re.match(pattern, initiative_and_revised_claim)
        try:
            claim_purpose = match_result.group(1)
            corrected_claim = match_result.group(2)
        except (IndexError, AttributeError):
            logger.error("replace_claim result cannot be parsed.")
            claim_purpose = ""
            corrected_claim = ""

        return claim_purpose, corrected_claim

    def _reply_using_combined_evidence(
            self,
            original_reply,
            object_dlg_history,
            last_user_utterance,
            evidences,
            system_parameters: dict,
    ):
        combined_reply = llm_generate(
            template_file=self.args.evidence_combiner_prompt,
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "last_user_utterance": last_user_utterance,
                "original_reply": original_reply,
                "evidences": evidences,
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=self.args.max_tokens,
            temperature=0,
            stop_tokens=None,
            top_p=self.args.top_p,
            frequency_penalty=self.args.frequency_penalty,
            presence_penalty=self.args.presence_penalty,
            postprocess=True,
            ban_line_break_start=True,
        )

        return combined_reply

    def _colbert_retrieve(
            self,
            query: str,
            num_paragraphs: int,
            rerank="none",
            num_paragraphs_for_reranking=0,
            top_p=1,
    ):
        """
        Args:
            `_num_paragraphs`: number of paragraphs that will be output
            `rerank` (str): one of 'none', 'recent' or a year like '2005' or 'voting'.
                'none' disables reranking. 'recent' retrieves more and returns the most recent ones.
                '2005' boosts the ranking of results that match 2005. The date of a result is determined by the year numbers it contains.
                'voting' ranks the results using the score of the document (i.e., voting in Stack Exchange)
            `top_p` (float): chooses from the smallest possible set of results whose cumulative probability exceeds top_p
        Returns:
            `passages` (list): a list of passage texts (excluding the title) with the highest similarities to the `query`
            `passage_scores` (list): a list of similarity scores of each passage in `passsages` with `query`
            `passage_titles` (list): a list of passage titles
        """

        if rerank != "none" and num_paragraphs_for_reranking < num_paragraphs:
            raise ValueError(
                "Need to retrieve at least as many paragraphs as num_paragraphs to be able to rerank."
            )

        if rerank == "none":
            num_paragraphs_for_reranking = num_paragraphs

        # print(self.colbert_endpoint, {'query': query, 'evi_num': num_paragraphs})
        response = requests.get(
            self.colbert_endpoint,
            json={"query": query, "evi_num": num_paragraphs_for_reranking},
        )
        if response.status_code != 200:
            raise Exception("ColBERT Search API Error: %s" % str(response))
        results = response.json()
        passages = []
        passage_titles = []
        for r in results["passages"]:
            r = r.split("|", maxsplit=1)
            passage_titles.append(r[0].strip())
            passages.append(r[1].strip())
        scores = results["passage_scores"]
        probs = results["passage_probs"]
        # print("probs = ", probs)
        top_p_cut_off = np.cumsum(probs) > top_p
        if not np.any(top_p_cut_off):
            # even if we include everything, we don't get to top_p
            top_p_cut_off = len(scores)
        else:
            top_p_cut_off = np.argmax(top_p_cut_off) + 1
        # print("top_p_cut_off = ", top_p_cut_off)
        passages, scores, passage_titles = (
            passages[:top_p_cut_off],
            scores[:top_p_cut_off],
            passage_titles[:top_p_cut_off],
        )

        if rerank == "none":
            pass
        elif rerank == "voting":  # Consider the collective intelligence on CQA platform.
            all_passage_votes = []
            for p in passages:
                voting = extract_vote(passage=p)
                all_passage_votes.append(voting)
            sort_fn = lambda x: x[3]
            passages, scores, passage_titles, all_passage_dates = list(
                zip(
                    *sorted(
                        zip(passages, scores, passage_titles, all_passage_votes),
                        reverse=True,
                        key=sort_fn,
                    )
                )
            )
        else:
            all_passage_dates = []
            for t, p in zip(passage_titles, passages):
                passage_years = extract_year(title=t, passage=p)
                all_passage_dates.append(passage_years)
            if rerank == "recent":
                sort_fn = lambda x: max(
                    x[3] if len(x[3]) > 0 else [0]
                )  # sort based on the latest year mentioned in the paragraph, demoting paragraphs that don't mention a year
            else:
                # rerank is a year
                try:
                    query_year = int(rerank)
                except ValueError as e:
                    # raise ValueError('rerank should be none, recent or an integer.')
                    logger.error(e)
                    return (
                        passages[:num_paragraphs],
                        scores[:num_paragraphs],
                        passage_titles[:num_paragraphs],
                    )
                sort_fn = lambda x: x[3].count(
                    query_year
                )  # boost the passages that have a matching year with the query, the more they mention the date the more we boost

            passages, scores, passage_titles, all_passage_dates = list(
                zip(
                    *sorted(
                        zip(passages, scores, passage_titles, all_passage_dates),
                        reverse=True,
                        key=sort_fn,
                    )
                )
            )

        # choose top num_paragraphs paragraphs
        passages, scores, passage_titles = (
            passages[:num_paragraphs],
            scores[:num_paragraphs],
            passage_titles[:num_paragraphs],
        )

        return passages, scores, passage_titles

    def _retrieve_evidences(self, claims, top_p: float = 1):
        """
        Retrieve evidences
        Args:
            - `claims` (list): list of (claim, year)
            - `top_p` (float): chooses from the smallest possible set of results whose cumulative probability exceeds top_p
        Returns:
            - `ret_output` (dict): a dict from claim_id to a list of `evidence`
                - each `evidence` is a list of length 5: [`title of wikipedia page`, _unused_, `wikipedia text`, _unused_, `similarity_score`]
        """
        ret_output = dict()
        for id, (cl, year) in enumerate(claims):
            if self.args.reranking_method == "none":
                passages, passage_scores, passage_titles = self._colbert_retrieve(
                    query=cl, num_paragraphs=self.evi_num, top_p=top_p
                )
            else:
                passages, passage_scores, passage_titles = self._colbert_retrieve(
                    query=cl,
                    num_paragraphs=self.evi_num,
                    rerank="voting",
                    num_paragraphs_for_reranking=6,
                    top_p=top_p,
                )
            evidences = []
            for passage, score, title in zip(passages, passage_scores, passage_titles):
                evidences.append([title, 0, passage, 0, score])
            # data =  json.dumps({"id": id, "claim": id_2_claim[str(id)], "evidence": evidences})
            # out.write(data + "\n")
            ret_output[id] = evidences

        return ret_output

    def _verify_claims(
            self,
            claims,
            ret_output,
            object_dlg_history,
            new_user_utterance,
            original_reply,
            system_parameters: dict,
            with_correct: bool = False,
    ):
        """
        Verify claims using retrieval output
        Args:
            - `claims` (list): list of (claim, year) pairs splitted
            - `ret_output` (dict): a dict from claim_id to a list of `evidence`
                - each `evidence` is a list of length 5: [`title of wikipedia page`, _unused_, `wikipedia text`, _unused_, `similarity_score`]
            - `object_dlg_history`(str): list of previous DialogueTurns
            - `last_user_utterance` (str): last user utterance
            - `original_reply`(str): GPT3 original response
            - `with_correct`(bool): reflect the system purpose and try to correct the claim with no additional cost
        Returns:
            - `ver_output` (list): a list of verification label ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") and the fixed claims
        """
        ver_output = []
        parameter_values_list = []

        for claim_id, (cl, year) in enumerate(claims):
            evidences = ret_output[claim_id][: self.evi_num]
            parameter_values_list.append(
                {
                    "dlg": object_dlg_history,
                    "last_user_utterance": new_user_utterance,
                    "original_reply": original_reply,
                    "claim": cl,
                    "evidence_titles": [e[0] for e in evidences],
                    "evidence_texts": [e[2] for e in evidences],
                }
            )

        # when using gold evidence, we do not split claim so claim is the same with original reply
        if self.args.skip_verification:
            all_verification_responses = ["is \"SUPPORTS\""] * len(claims)
        elif with_correct:
            all_verification_responses = llm_generate(
                template_file=self.args.verification_prompt,
                prompt_parameter_values=parameter_values_list,
                engine=system_parameters.get("engine", self.args.engine),
                max_tokens=200,
                temperature=0,
                stop_tokens=None,
                postprocess=False,
                ban_line_break_start=True,
            )
        else:
            all_verification_responses = llm_generate(
                template_file=self.args.verification_prompt,
                prompt_parameter_values=parameter_values_list,
                engine=system_parameters.get("engine", self.args.engine),
                max_tokens=200,
                temperature=0,
                stop_tokens=None,
                postprocess=False,
                ban_line_break_start=True,
            )

        claim_purpose = ""
        fixed_claim = ""
        for (cl, year), verification_response in zip(
                claims, all_verification_responses
        ):
            # the following handles cases where smaller models like gpt-35-turbo do not follow the few-shot examples' format
            if (
                    'is "supports"' in verification_response.lower()
                    or "no fact-checking is needed for this claim"
                    in verification_response.lower()
                    or "the fact-checking result is not applicable to this response"
                    in verification_response.lower()
            ):
                verification_label = "SUPPORTS"
                fixed_claim = cl
            else:
                if (
                        'the fact-checking result is "not enough info"'
                        in verification_response.lower()
                ):
                    verification_label = "NOT ENOUGH INFO"
                else:
                    verification_label = "REFUTES"  # default set to be "REFUTES"

                if with_correct:
                    pattern = r'.*\nThe purpose of this claim: "(.*)"\nThe supported claim for this purpose: "(.*)"'
                    match_result = re.match(pattern, verification_response)
                    try:
                        claim_purpose = match_result.group(1)
                        fixed_claim = match_result.group(2)
                        logger.info(f"Original unsupported claim: {cl}")
                        logger.info(f"Claim purpose: {claim_purpose}")
                        logger.info(f"Corrected claim: {fixed_claim}")
                        if "NO IDEA" in fixed_claim:
                            fixed_claim = ""
                    except (IndexError, AttributeError):
                        logger.error("replace_claim result cannot be parsed.")

            ver_output.append({"label": verification_label, "claim_purpose": claim_purpose, "fixed_claim": fixed_claim})

        return ver_output

    def _remove_claims_from_previous_turns(self, claims: List, object_dlg_history):
        """
        Removes claims that are repeated from the last turn. This is often the result of LLM making a mistake while splitting claims.
        But even if it is actually a claim that the chatbot repeats, removing it here is beneficial as it will reduce repetitiveness.
        """
        previous_turn_claims = []
        for i in range(len(object_dlg_history)):
            previous_turn_claims.extend([c[0] for c in object_dlg_history[i].claims])
        claims = [c for c in claims if c[0] not in previous_turn_claims]

        return claims
