"""
Functionality to work with .prompt files
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import List, Union
import copy
import openai
import logging
from typing import List
import openai
from openai import OpenAIError
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined
import pytz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import threading
from functools import lru_cache, partial, wraps
import requests
from text_generation import Client

logger = logging.getLogger(__name__)

# singleton
jinja_environment = Environment(
    loader=FileSystemLoader(["./", "./pipelines/"]),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix="#",
    undefined=StrictUndefined,
)  # StrictUndefined raises an exception if a variable in template is not passed to render(), instead of setting it as empty

# define chat tags for chat models
system_start = "<|system_start|>"
system_end = "<|system_end|>"
user_start = "<|user_start|>"
user_end = "<|user_end|>"
assistant_start = "<|assistant_start|>"
assistant_end = "<|assistant_end|>"

azure_api_version = "2023-05-15"
all_openai_resources = [
    {
        "api_type": "azure",
        "api_version": azure_api_version,
        "api_base": "https://ovalopenairesource.openai.azure.com/",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "engine_map": {
            "text-davinci-003": "text-davinci-003",
            "gpt-35-turbo": "gpt-35-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-32k": "gpt-4-32k",
        },
    },
    {
        "api_type": "azure",
        "api_version": azure_api_version,
        "api_base": "https://reactgenie-dev.openai.azure.com/",
        "api_key": os.getenv("OPENAI_API_KEY_1"),
        "engine_map": {
            "text-davinci-003": "text_user_study",
            "gpt-35-turbo": "test_35",
            "gpt-4": "test",
            "gpt-4-32k": "test2",
        },
    },
    {
        "api_type": "azure",
        "api_version": azure_api_version,
        "api_base": "https://wikidata.openai.azure.com/",
        "api_key": os.getenv("OPENAI_API_KEY_2"),
        "engine_map": {
            "text-davinci-003": "text-davinci-003",
            "gpt-35-turbo": "gpt-35-turbo",
            "gpt-4": "gpt4-8k-playground",
        },
    },
    {
        "api_type": "azure",
        "api_version": azure_api_version,
        "api_base": "https://oval-france-central.openai.azure.com/",
        "api_key": os.getenv("OPENAI_API_KEY_3"),
        "engine_map": {
            "gpt-35-turbo": "gpt-35-turbo",
            "gpt-4": "gpt-4",
        },
    },
]
all_openai_resources = [
    a for a in all_openai_resources if a["api_key"] is not None
]  # remove resources for which we don't have a key

# only used when other resources fail due to Azure's content filter
backup_openai_resource = all_openai_resources[
    0
]  # filtering is turned off for this Azure OpenAI resource
# backup_openai_resource = {
# "api_type": "open_ai",
# "api_version": None,
# "api_base": "https://api.openai.com/v1",
# "api_key": os.getenv("OPENAI_API_KEY_BACKUP"),
# "engine_map": {
#     "text-davinci-003": "text-davinci-003",
#     "gpt-35-turbo": "gpt-3.5-turbo-0301",
#     "gpt-4": "gpt-4-0314",
# },
# }

# for prompt debugging
prompt_log_file = "data/prompt_logs.json"
debug_prompts = False
prompt_logs = []
prompts_to_skip_for_debugging = [
    "benchmark/prompts/user_with_passage.prompt",
    "benchmark/prompts/user_with_topic.prompt",
]

# for local engines
local_engine_endpoint = None
local_engine_prompt_format = None
local_engine_client = None


def set_debug_mode():
    global debug_prompts
    debug_prompts = True


def set_local_engine_properties(engine_endpoint: str, prompt_format: str):
    global local_engine_endpoint
    local_engine_endpoint = engine_endpoint
    global local_engine_prompt_format
    local_engine_prompt_format = prompt_format
    global local_engine_client
    local_engine_client = OpenAILikeClient(timeout=10, base_url=local_engine_endpoint)


cost_lock = threading.Lock()

local_model_list = ["llama"]
chat_models_list = ["gpt-35-turbo", "gpt-4-32k", "gpt-4"]
inference_cost_per_1000_tokens = {
    "ada": (0.0004, 0.0004),
    "babbage": (0.0005, 0.0005),
    "curie": (0.002, 0.002),
    "davinci": (0.02, 0.02),
    "gpt-35-turbo": (0.002, 0.002),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4": (0.03, 0.06),
}  # (prompt, completion)

total_cost = 0  # in USD


class OpenAILikeClient(Client):
    def __init__(
        self,
        base_url,
        headers=None,
        cookies=None,
        timeout=10,
    ):
        super().__init__(base_url, headers, cookies, timeout)

    def generate(
        self,
        prompt,
        max_tokens,
        stop,
        temperature,
        repetition_penalty=None,
        top_k=None,
        top_p=None,
    ):
        if temperature == 0:
            temperature = 1
            do_sample = False
        else:
            do_sample = True

        if top_p == 1:
            top_p = None

        return super().generate(
            prompt,
            do_sample,
            max_tokens,
            best_of=None,
            repetition_penalty=repetition_penalty,
            return_full_text=False,
            seed=None,
            stop_sequences=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=None,
            typical_p=None,
            watermark=False,
            decoder_input_details=False,
        )


def add_to_total_cost(amount: float):
    global total_cost
    with cost_lock:
        total_cost += amount


def get_total_cost():
    global total_cost
    return total_cost


def _model_name_to_cost(model_name: str) -> float:
    for model_family in inference_cost_per_1000_tokens.keys():
        if model_family in model_name:
            return inference_cost_per_1000_tokens[model_family]
    raise ValueError("Did not recognize OpenAI model name %s" % model_name)


def write_prompt_logs_to_file():
    global prompt_logs
    with open(prompt_log_file, "w") as f:
        f.write(json.dumps(prompt_logs, indent=4, ensure_ascii=False))


def _remove_starting_and_ending_whitespace(text):
    # remove whitespace at the beginning and end of each line
    return "\n".join([line.strip() for line in text.split("\n")])


def conditional_lru_cache(condition_func, maxsize):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(func)
        def wrapper(**kwargs):
            if condition_func(**kwargs):
                return cached_func(**kwargs)
            else:
                return func(**kwargs)

        return wrapper

    return decorator


def cache_if_temperature_is_zero(**kwargs):
    c = kwargs.get("temperature") == 0
    # print('temparature == 0: ', c)
    return c


def _openai_completion_with_backup(original_engine_name: str, kwargs):
    if original_engine_name in chat_models_list:
        # These models use ChatCompletion API, which doesn't support batch inference, so we use multithreading instead.
        # We use multithreading (instead of multiprocessing) because this method is I/O-bound, mostly waiting for an HTTP response to come back.

        engine = kwargs["engine"]
        prompt = kwargs["prompt"]
        max_tokens = kwargs["max_tokens"]
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        frequency_penalty = kwargs["frequency_penalty"]
        presence_penalty = kwargs["presence_penalty"]
        stop = kwargs["stop"]
        logit_bias = kwargs.get("logit_bias", {})

        if kwargs["api_type"] == "azure":
            f = partial(
                openai.ChatCompletion.create,
                engine=engine,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                logit_bias=logit_bias,
                api_base=kwargs["api_base"],
                api_key=kwargs["api_key"],
                api_type=kwargs["api_type"],
                api_version=kwargs["api_version"],
            )
        else:
            # use 'model' instead of 'engine'. no `api_version``
            f = partial(
                openai.ChatCompletion.create,
                model=engine,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                logit_bias=logit_bias,
                api_base=kwargs["api_base"],
                api_key=kwargs["api_key"],
                api_type=kwargs["api_type"],
            )

        prompt = _convert_filled_prompt_to_chat_messages(prompt)
        # print("prompt = ", json.dumps(prompt, indent=2, ensure_ascii=False))

        with ThreadPoolExecutor(len(prompt)) as executor:
            thread_outputs = [executor.submit(f, messages=p) for p in prompt]
        thread_outputs = [o.result() for o in thread_outputs]

        # make the outputs of these multiple calls to openai look like one call with batching
        ret = thread_outputs[0]
        for i, o in enumerate(thread_outputs[1:]):
            o["choices"][0]["index"] = i + 1
            ret["choices"].append(o["choices"][0])
            ret["usage"]["completion_tokens"] += o["usage"]["completion_tokens"]
            ret["usage"]["prompt_tokens"] += o["usage"]["prompt_tokens"]
            ret["usage"]["total_tokens"] += o["usage"]["total_tokens"]

        for c in ret["choices"]:
            # key 'content' does not exist for gpt-4 when output is empty. Example input: "who plays whitey bulger's girlfriend in black mass"
            if c["finish_reason"] == "content_filter":
                if (
                    backup_openai_resource is not None
                    and backup_openai_resource["api_key"] is not None
                ):
                    # raise exception so that the behavior is like non-chat OpenAI models
                    raise OpenAIError("Azure OpenAI’s content management policy")
                else:
                    c["text"] = None
            else:
                c["text"] = c["message"]["content"]
    elif original_engine_name in local_model_list:
        # models like LLaMA
        # print("kwargs = ", kwargs)
        # req = requests.post(
        #     url=local_engine_endpoint,
        #     json={
        #         **kwargs
        #     })
        # ret = req.json()
        max_tokens = kwargs["max_tokens"]
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        frequency_penalty = kwargs["frequency_penalty"]
        presence_penalty = kwargs["presence_penalty"]
        stop = kwargs["stop"]

        if (
            "presence_penalty" in kwargs
            and kwargs["presence_penalty"] is not None
            and kwargs["presence_penalty"] != 0
        ):
            logger.warning(
                "Ignoring `presence_penalty` since it is not supported by this model."
            )
        if (
            "frequency_penalty" in kwargs
            and kwargs["frequency_penalty"] is not None
            and kwargs["frequency_penalty"] != 0
        ):
            logger.warning(
                "Ignoring `frequency_penalty` since it is not supported by this model."
            )

        with ThreadPoolExecutor(len(kwargs["prompt"])) as executor:
            thread_outputs = [
                executor.submit(
                    local_engine_client.generate,
                    prompt=p,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                for p in kwargs["prompt"]
            ]
        thread_outputs = [o.result().generated_text for o in thread_outputs]
        # print(thread_outputs)
        ret = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": i,
                    "logprobs": None,
                    "text": thread_outputs[i],
                }
                for i in range(len(thread_outputs))
            ],
            "created": 0,
            "id": "",
            "model": original_engine_name,
            "object": "text_completion",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        # print("ret = ", ret)
    else:
        # models like text-davinci-003
        if kwargs["api_type"] == "open_ai":
            kwargs["model"] = kwargs["engine"]
            del kwargs["engine"]
            del kwargs["api_version"]
        del kwargs["prompt_role"]
        # print("api_base = ", kwargs["api_base"])
        ret = openai.Completion.create(**kwargs)

        for c in ret["choices"]:
            if c["finish_reason"] == "content_filter":
                if (
                    backup_openai_resource is not None
                    and backup_openai_resource["api_key"] is not None
                ):
                    # raise exception so that the behavior is like non-chat OpenAI models
                    raise OpenAIError("Azure OpenAI’s content management policy")

        # return it back
        # if kwargs["api_type"] == 'open_ai':
        #     kwargs["engine"] = kwargs["model"]
        #     del kwargs["model"]

        # output looks like
        #  "choices": [
        #         {
        #         "finish_reason": "stop",
        #         "index": 0,
        #         "logprobs": null,
        #         "text": " No.]"
        #         }
        #     ],
        #     "created": 1682393599,
        #     "id": "cmpl-793iZfhYTA2pAHA3LJSGmy6aUsQO7",
        #     "model": "text-davinci-003",
        #     "object": "text_completion",
        #     "usage": {
        #         "completion_tokens": 2,
        #         "prompt_tokens": 937,
        #         "total_tokens": 939
        #     }
        #     }

        # reorder outputs to have the same order as inputs
        choices_in_correct_order = [{}] * len(ret.choices)
        for choice in ret.choices:
            choices_in_correct_order[choice.index] = choice
        ret.choices = choices_in_correct_order

    return ret


def _set_openai_fields(kwargs, openai_resource):
    kwargs_copy = copy.deepcopy(kwargs)
    kwargs_copy["api_type"] = openai_resource["api_type"]
    kwargs_copy["api_version"] = openai_resource["api_version"]
    kwargs_copy["api_base"] = openai_resource["api_base"]
    kwargs_copy["api_key"] = openai_resource["api_key"]
    kwargs_copy["engine"] = openai_resource["engine_map"][kwargs["engine"]]

    # print("setting api_base to ", kwargs_copy["api_base"])

    return kwargs_copy


@conditional_lru_cache(cache_if_temperature_is_zero, maxsize=10000)
def _openai_completion_with_caching(**kwargs):
    original_engine_name = kwargs["engine"]

    try:
        if original_engine_name not in local_model_list:
            # decide which Azure resource to send this request to.
            # Use hash so that each time this function gets called with the same parameters after a backoff, the request gets sent to the same resource
            potential_openai_resources = [
                resource
                for resource in all_openai_resources
                if kwargs["engine"] in resource["engine_map"]
            ]
            openai_resource = potential_openai_resources[
                hash(json.dumps(kwargs, sort_keys=True).encode())
                % len(potential_openai_resources)
            ]
            # uniform load balancing instead of hashing
            # openai_resource = potential_openai_resources[random.randrange(len(potential_openai_resources))]
            ret = _openai_completion_with_backup(
                original_engine_name, _set_openai_fields(kwargs, openai_resource)
            )
        else:
            ret = _openai_completion_with_backup(original_engine_name, kwargs)
    except OpenAIError as e:
        if "Azure OpenAI’s content management policy" in str(e):
            if (
                backup_openai_resource is not None
                and backup_openai_resource["api_key"] is not None
            ):
                logger.info(
                    "Azure content filter triggered, using backup OpenAI resource."
                )
                ret = _openai_completion_with_backup(
                    original_engine_name,
                    _set_openai_fields(kwargs, backup_openai_resource),
                )
            else:
                logger.error("No output due to Azure's content filtering.")
                # logger.error(str(kwargs['prompt']))
                return {
                    "choices": [
                        {
                            "finish_reason": "content_filter",
                            "index": i,
                            "logprobs": None,
                            "text": None,
                        }
                        for i in range(len(kwargs["prompt"]))
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }
        else:
            raise e

    # calculate the cost
    # print('ret = ', ret)
    if original_engine_name not in local_model_list:
        cost_prompt, cost_completion = _model_name_to_cost(original_engine_name)
        total_cost = (
            ret["usage"]["prompt_tokens"] * cost_prompt
            + ret["usage"].get("completion_tokens", 0) * cost_completion
        ) / 1000
        add_to_total_cost(total_cost)

    return ret


@retry(
    retry=retry_if_exception_type(OpenAIError),
    wait=wait_exponential(min=1, max=120, exp_base=2),
    stop=stop_after_attempt(7),
    #    before_sleep=before_sleep_log(logger, logging.ERROR)
)
def _openai_completion_with_backoff(**kwargs):
    if kwargs["temperature"] == 0:
        # Convert the inputs to be hashable. Necessary for caching to work.
        for a in kwargs:
            if isinstance(kwargs[a], list):
                kwargs[a] = tuple(kwargs[a])

    return _openai_completion_with_caching(**kwargs)


def _fill_template(template_file, prompt_parameter_values, get_rendered_blocks=False):
    # logger.info("Filling template %s", template_file)
    template = jinja_environment.get_template(template_file)

    prompt_parameter_values["system_start"] = system_start
    prompt_parameter_values["system_end"] = system_end
    prompt_parameter_values["user_start"] = user_start
    prompt_parameter_values["user_end"] = user_end
    prompt_parameter_values["assistant_start"] = assistant_start
    prompt_parameter_values["assistant_end"] = assistant_end

    # always make these useful constants available in a template
    # make a new function call each time since the date might change during a long-term server deployment
    today = datetime.now(pytz.timezone("US/Pacific")).date()
    prompt_parameter_values["today"] = today.strftime("%B %d, %Y")  # May 30, 2023
    prompt_parameter_values["current_year"] = today.year
    prompt_parameter_values["location"] = "the U.S."
    prompt_parameter_values["chatbot_name"] = "StackExchangeInterface"

    # TODO(yijia): hardcode at present.
    prompt_parameter_values["domain"] = "https://cooking.stackexchange.com"

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = _remove_starting_and_ending_whitespace(filled_prompt)

    # Access the 'content' block and render it
    rendered_blocks = {}
    if get_rendered_blocks:
        for block_name in template.blocks.keys():
            block = template.blocks[block_name](
                template.new_context(vars=prompt_parameter_values)
            )
            rendered = "".join(block)
            rendered = remove_chat_tags(
                rendered
            )  # blocks are used for logging and local engines, so should never have chat tags
            rendered_blocks[block_name] = rendered

    return filled_prompt, rendered_blocks


def _ban_line_break_start_generate(
    filled_prompt: List[str],
    engine,
    max_tokens,
    temperature,
    stop_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    prompt_role,
):
    no_line_break_length = 3
    # generate 3 tokens that definitely are not line_breaks
    no_line_break_start = _openai_completion_with_backoff(
        engine=engine,
        prompt=filled_prompt,
        max_tokens=no_line_break_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_tokens,
        # \n, \n\n
        logit_bias={"198": -100, "628": -100},
        prompt_role=prompt_role,
    )
    # print(no_line_break_start)
    outputs = []
    second_round_idxs = []
    second_round_prompts = []
    for idx, o in enumerate(no_line_break_start["choices"]):
        stop_reason = o["finish_reason"]
        print("stop_reason = ", stop_reason)
        text = o["text"]
        outputs.append(text)

        if stop_reason == "stop":
            # this short generation already generated the stop_token
            continue

        assert stop_reason == "length"
        second_round_idxs.append(idx)
        second_round_prompts.append(filled_prompt[idx] + text)

    # print('filled_prompt = ', filled_prompt)
    # print('second_round_prompts = ', second_round_prompts)
    # for prompts that have not hit stop_token yet
    generation_output = _openai_completion_with_backoff(
        engine=engine,
        prompt=second_round_prompts,
        max_tokens=max_tokens - no_line_break_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_tokens,
        prompt_role=prompt_role,
    )
    # print(generation_output)
    for i, original_index in enumerate(second_round_idxs):
        outputs[original_index] = (
            outputs[original_index] + generation_output["choices"][i]["text"]
        )

    return outputs


def _generate(
    filled_prompt: List[str],
    engine,
    max_tokens,
    temperature,
    stop_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    postprocess,
    ban_line_break_start,
    prompt_role,
) -> List[str]:
    # logger.info('LLM input = %s', json.dumps(filled_prompt, indent=2))
    if not ban_line_break_start and stop_tokens is not None and "\n" in stop_tokens:
        logger.warning(
            'Consider setting ban_line_break=True when "\n" is in stop_tokens'
        )

    # ChatGPT and GPT-4 use ChatCompletion and a different format from older models.
    # The following is a naive implementation of few-shot prompting, which may be improved.
    # if engine in chat_models_list:
    # ban_line_break_start = False  # no need to prevent new lines in chat models

    if engine in local_model_list:
        # this feature is not supported for local LLMs
        ban_line_break_start = False

    # outputs = dict([(i, "")] for i in range(len(filled_prompt)))

    generation_output = _openai_completion_with_backoff(
        engine=engine,
        prompt=filled_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_tokens,
        prompt_role=prompt_role,
    )

    outputs = []
    for choice in generation_output["choices"]:
        if choice["text"]:
            outputs.append(choice["text"])

    if ban_line_break_start:
        # TODO we can just regenerate the ones that start with newline, instead of everything, but this is really rare
        for o in outputs:
            if o.startswith("\n"):
                logger.info("Regenerating due to unwanted \\n")
                outputs = _ban_line_break_start_generate(
                    filled_prompt=filled_prompt,
                    engine=engine,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_tokens=stop_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    prompt_role=prompt_role,
                )
                break

    outputs = [o.strip() for o in outputs]
    if postprocess:
        outputs = [_postprocess_generations(o) for o in outputs]

    return outputs


def _convert_filled_prompt_to_chat_messages(filled_prompt: List[str]):
    ret = []
    # print("before conversion", json.dumps(filled_prompt, indent=2))
    for fp in filled_prompt:
        # TODO check that start system is unique
        messages = []

        system_s = fp.find(system_start)
        system_e = fp.find(system_end, system_s)
        if system_s < 0:
            # did not find a system message in the prompt, so will put everything inside system for backward compatibility
            messages.append(
                {
                    "role": "system",
                    "content": fp.strip(),
                }
            )
            ret.append(messages)
            continue

        messages.append(
            {
                "role": "system",
                "content": fp[system_s + len(system_start) : system_e].strip(),
            }
        )

        last_index = 0
        while True:
            user_s = fp.find(user_start, last_index)
            assistant_s = fp.find(assistant_start, last_index)
            if (
                user_s >= 0
                and assistant_s < 0
                or (user_s >= 0 and user_s < assistant_s)
            ):
                user_e = fp.find(user_end, user_s)
                assert user_e >= 0, "Missing closing tag for user"
                last_index = user_e
                messages.append(
                    {
                        "role": "user",
                        "content": fp[user_s + len(user_start) : user_e].strip(),
                    }
                )
            elif (
                user_s < 0
                and assistant_s >= 0
                or (assistant_s >= 0 and user_s > assistant_s)
            ):
                assistant_e = fp.find(assistant_end, assistant_s)
                assert assistant_e >= 0, "Missing closing tag for assistant"
                last_index = assistant_e
                messages.append(
                    {
                        "role": "assistant",
                        "content": fp[
                            assistant_s + len(assistant_start) : assistant_e
                        ].strip(),
                    }
                )
            else:
                assert user_s < 0 and assistant_s < 0
                break
        ret.append(messages)

    return ret


def remove_chat_tags(s: str):
    return _remove_starting_and_ending_whitespace(
        s.replace(system_start, "")
        .replace(system_end, "")
        .replace(user_start, "")
        .replace(user_end, "")
        .replace(assistant_start, "")
        .replace(assistant_end, "")
    )  # need to remove starting and ending whitespace because after removing chat tags, whitespace that used to be in the middle might become starting or ending whitespace


def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = " ".join(generation_output.split())
    # print("generation_output = ", generation_output)

    original_generation_output = generation_output
    # remove extra dialog turns, if any
    turn_indicators = [
        "You:",
        "They:",
        "Context:",
        "You said:",
        "They said:",
        "Assistant:",
        "Chatbot:",
    ]
    for t in turn_indicators:
        while generation_output.find(t) > 0:
            generation_output = generation_output[: generation_output.find(t)]

    generation_output = generation_output.strip()
    # delete half sentences
    if len(generation_output) == 0:
        logger.error(
            "LLM output is empty after postprocessing. Before postprocessing it was %s",
            original_generation_output,
        )
        return generation_output

    if generation_output[-1] not in {".", "!", "?"} and generation_output[-2:] != '."':
        # handle preiod inside closing quotation
        last_sentence_end = max(
            generation_output.rfind("."),
            generation_output.rfind("!"),
            generation_output.rfind("?"),
            generation_output.rfind('."') + 1,
        )
        if last_sentence_end > 0:
            generation_output = generation_output[: last_sentence_end + 1]
    return generation_output


def llm_generate(
    template_file: str,
    prompt_parameter_values: Union[dict, List[dict]],
    engine: str,
    max_tokens: int,
    temperature: float,
    stop_tokens,
    top_p: float = 0.9,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    postprocess: bool = True,
    ban_line_break_start: bool = False,
    filled_prompt=None,
    prompt_role="system",
):
    """
    Generates continuations for one or more prompts in parallel
    Inputs:
        prompt_parameter_values: dict or list of dict. If the input is a list, the output will be a list as well
        filled_prompt: gives direct access to the underlying model, without having to load a prompt template from a .prompt file. Used for testing.
        ban_line_break_start: can in the worst case triple the cost, though in practice (and especially with good prompts) this only happens for a small fraction of inputs
        prompt_role: The role to use for inputting the prompt. "system", "assistant" or "user". Only used for OpenAI's chat models
    """
    if not (
        filled_prompt is None
        and prompt_parameter_values is not None
        and template_file is not None
    ) and not (
        filled_prompt is not None
        and prompt_parameter_values is None
        and template_file is None
    ):
        raise ValueError(
            "Can only use filled_prompt if template_file and prompt_parameter_values are None"
        )

    # convert to a single element list so that the rest of the code only has to deal with a list
    input_was_list = True
    if filled_prompt is None:
        assert prompt_parameter_values is not None
        if not isinstance(prompt_parameter_values, list):
            input_was_list = False
            prompt_parameter_values = [prompt_parameter_values]
        filled_prompt = [
            _fill_template(template_file, p, get_rendered_blocks=True)
            for p in prompt_parameter_values
        ]
        # convert list of tuples to tuple of lists
        filled_prompt, rendered_blocks = tuple(zip(*filled_prompt))
        filled_prompt, rendered_blocks = list(filled_prompt), list(rendered_blocks)

        # remove short_instruction blocks
        for idx, block in enumerate(rendered_blocks):
            if "short_instruction" in block:
                filled_prompt[idx] = filled_prompt[idx].replace(
                    block["short_instruction"], ""
                )

        if engine in local_model_list:
            if local_engine_prompt_format == "none":
                pass
            elif local_engine_prompt_format == "alpaca":
                filled_prompt = [
                    (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    ).format_map(block)
                    for block in rendered_blocks
                ]
            elif local_engine_prompt_format == "simple":
                filled_prompt = [
                    "{instruction}\n\n{input}\n".format_map(block)
                    for block in rendered_blocks
                ]
            else:
                raise ValueError(
                    "Unknown prompt format specified for the local engine."
                )
        if engine not in chat_models_list:
            # print("rendered_blocks = ", rendered_blocks)
            filled_prompt = [remove_chat_tags(f) for f in filled_prompt]
    else:
        if not isinstance(filled_prompt, list):
            input_was_list = False
            filled_prompt = [filled_prompt]

    assert isinstance(filled_prompt, list)

    outputs = _generate(
        filled_prompt,
        engine,
        max_tokens,
        temperature,
        stop_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        postprocess,
        ban_line_break_start,
        prompt_role,
    )

    global debug_prompts
    global prompt_logs
    if debug_prompts:
        for i, o in enumerate(outputs):
            if template_file in prompts_to_skip_for_debugging:
                continue
            prompt_logs.append(
                {
                    "template_name": template_file,
                    "instruction": rendered_blocks[i]["short_instruction"]
                    if "short_instruction" in rendered_blocks[i]
                    else rendered_blocks[i]["instruction"],
                    "input": rendered_blocks[i]["input"],
                    "output": o,
                }
            )

    if outputs == []:
        outputs = ""
    if len(outputs) == 1 and not input_was_list:
        # convert back to a single item
        outputs = outputs[0]
    return outputs
