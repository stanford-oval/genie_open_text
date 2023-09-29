"""
Chat with GPT-3, with retrieval and fact checking at each dialog turn.
"""
import logging
from typing import List
import argparse
import json
import readline  # enables keyboard arrows when typing in the terminal

from pipelines.dialog_turn import DialogueTurn
from pipelines.chatbot import Chatbot
from pipelines.utils import input_user, print_chatbot, make_parent_directories
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.llm import set_debug_mode, write_prompt_logs_to_file

logging.getLogger("openai").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


def main(args):
    chatbot = Chatbot(args)

    dlg_history: List[DialogueTurn] = []

    while True:
        try:
            user_utterance = input_user()
        except EOFError:
            # stop the chatbot
            break

        # check for special commands
        if user_utterance in args.quit_commands:
            # stop the chatbot
            break
        if user_utterance in ["clear", "cls"]:
            # restart the dialog
            dlg_history = []
            continue

        new_dlg_turn = chatbot.generate_next_turn(
            dlg_history, user_utterance, pipeline=args.pipeline
        )

        dlg_history.append(new_dlg_turn)
        turn_log = json.dumps(new_dlg_turn.log(), indent=2, ensure_ascii=False)
        logger.info("Turn log: %s", turn_log)
        print_chatbot("Chatbot: " + new_dlg_turn.agent_utterance)

        make_parent_directories(args.output_file)
        with open(args.output_file, "a") as outfile:
            if len(dlg_history) == 1:
                # first turn
                outfile.write("=====\n")

            outfile.write("User: " + new_dlg_turn.user_utterance + "\n")
            outfile.write("Chatbot: " + new_dlg_turn.agent_utterance + "\n")

        with open(args.output_file.strip("txt") + "log", "a") as outfile:
            outfile.write(turn_log)
            outfile.write("\n")


if __name__ == "__main__":
    # text generation arguments
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to write the outputs."
    )
    parser.add_argument("--no_logging", action="store_true", help="Disables logging")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Write prompts inputs/outputs to a file for debugging",
    )

    parser.add_argument(
        "--quit_commands",
        type=str,
        default=["quit", "q", "Exit"],
        help="The conversation will continue until this string is typed in.",
    )

    args = parser.parse_args()
    check_pipeline_arguments(args)

    if args.no_logging:
        logging.basicConfig(
            level=logging.ERROR, format=" %(name)s : %(levelname)-8s : %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
        )

    if args.debug_mode:
        set_debug_mode()

    main(args)

    if args.debug_mode:
        write_prompt_logs_to_file()
