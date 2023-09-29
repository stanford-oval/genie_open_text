"""
Add the common input arguments
"""

from pipelines.llm import local_model_list, set_local_engine_properties


def add_pipeline_arguments(parser):
    # determine components of the pipeline
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=[
            "generate",
            "genie"
        ],
        default="verify_and_correct",
        help="The type of pipeline used to imrpove GPT-3 response. Only used to know which modules to load.",
    )
    parser.add_argument(
        "--claim_prompt_template_file",
        type=str,
        default="prompts/split_claim_rewrite.prompt",
        help="The path to the file containing the claim LLM prompt.",
    )
    parser.add_argument(
        "--summary_prompt",
        type=str,
        default="prompts/single_summary.prompt",
        help="What prompt to use to summarize retrieved passage into bullet points."
    )
    parser.add_argument(
        "--initial_search_prompt",
        type=str,
        default="prompts/initial_search.prompt",
        help="What prompt to use to get initial search query from the user input."
    )
    parser.add_argument(
        "--verification_prompt",
        type=str,
        default="prompts/verify_and_correct.prompt",
        help="What prompt to use to verify (or with correction) the LLM's output claim."
    )
    parser.add_argument(
        "--evidence_combiner_prompt",
        type=str,
        default="prompts/evidence_combiner.prompt",
        help="What prompt to use to combine bullet points into the final output."
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="If True, all claims will be considered correct without fact-checking. Especially useful to speed up debugging of the other parts of the pipeline.",
    )

    parser.add_argument(
        "--colbert_endpoint",
        type=str,
        default="http://127.0.0.1:5000/search",
        help="whether using colbert for retrieval.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=[
            "atlas",
            "gpt-35-turbo",
            "text-davinci-001",
            "text-davinci-002",
            "text-davinci-003",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-0314",
        ]
        + local_model_list,
        help="The LLM engine to use.",
    )  # choices are from the smallest to the largest model
    parser.add_argument(
        "--local_engine_endpoint",
        type=str,
        help="Need to be provided if --engine is a local LLM",
    )
    parser.add_argument(
        "--local_engine_prompt_format",
        type=str,
        choices=["none", "alpaca", "simple"],
        help="Need to be provided if --engine is a local LLM",
    )
    parser.add_argument(
        "--reranking_method",
        type=str,
        choices=["none", "date", "voting"],
        default="none",
        help="Only used for retrieve_and_generate pipeline",
    )

    # LLM generation hyperparameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        required=False,
        help="Only affects user-facing prompts",
    )

    parser.add_argument(
        "--evi_num",
        type=int,
        default=5,
        help="Number of evidences to retrieve per claim.",
    )


def check_pipeline_arguments(args):
    # make sure for ATLAS, both engine and pipeline are set to 'atlas'
    if hasattr(args, "pipeline"):
        if (args.engine == "atlas" and args.pipeline != "atlas") or (
            args.engine != "atlas" and args.pipeline == "atlas"
        ):
            raise ValueError(
                "When using ATLAS, both `engine` and `pipeline` input arguments should be set to 'atlas'."
            )

    if args.engine in local_model_list:
        if (
            args.local_engine_endpoint is None
            or args.local_engine_prompt_format is None
        ):
            raise ValueError(
                "Need to provide a local engine endpoint and prompt format for local LLMs."
            )
    if (
        args.local_engine_endpoint is not None
        and args.local_engine_prompt_format is not None
    ):
        # if these parameters are provided, set them anyways. This helps with back-end server, where the engine might change to a local one on the fly.
        set_local_engine_properties(
            args.local_engine_endpoint, args.local_engine_prompt_format
        )
