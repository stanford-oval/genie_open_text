import logging
from typing import List
from .llm import llm_generate

logger = logging.getLogger(__name__)


class ClaimSplitter:
    def __init__(self, args):
        if not args.claim_prompt_template_file:
            self.prompt_template_file = "prompts/split_claim_rewrite.prompt"
        else:
            self.prompt_template_file = args.claim_prompt_template_file
        self.args = args

    def split_claim(
        self,
        dialog_history: List,
        new_user_utterance: str,
        current_agent_utterance: str,
        system_parameters: dict,
        dialog_topic: str = None,
    ):
        """
        dialog_topic: used for splitting claims of a simulated dialog we want to evaluate
        """
        claims_output = llm_generate(
            template_file=self.prompt_template_file,
            prompt_parameter_values={
                "dlg": dialog_history,
                "new_user_utterance": new_user_utterance,
                "current_agent_utterance": current_agent_utterance,
                "dialog_topic": dialog_topic,
            },
            engine=system_parameters.get("engine", self.args.engine),
            max_tokens=300,
            temperature=0,
            stop_tokens=["====="],
            postprocess=False,
        )

        if claims_output.startswith("Yes. "):
            # necessary for distilled models
            claims_output = claims_output[5:]
        all_claims = self._format_output(claims_output)

        return all_claims

    def _format_output(self, output):
        lines = output.split("\n")
        if lines[0].startswith("Nothing."):
            # no claims detected
            return []
        all_claims = []
        try:
            for c in lines:
                claim = c
                cleaned_claim = claim.split(f"- ")[-1].strip()
                if cleaned_claim:
                    split_term = " The year of the results is "
                    if split_term not in cleaned_claim:
                        split_term = " The year of the claim is "
                    splitted = cleaned_claim.split(split_term)
                    if len(splitted) == 2:
                        cleaned_claim, year = splitted
                        year = year[1:-2]
                    else:
                        # sometimes model output may not be well-formatted (e.g. N/A); default to none
                        cleaned_claim = splitted[0]
                        year = "none"
                    all_claims.append((cleaned_claim, year))
        except Exception as e:
            logger.error("Error while parsing claims in %s: %s", output, str(e))
            raise e

        return all_claims
