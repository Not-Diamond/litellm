import types
import requests
from typing import Callable, Optional, Dict, List
import httpx

import litellm
from litellm.utils import ModelResponse


# dict to map notdiamond providers and models to litellm providers and models
notdiamond2litellm = {
    # openai
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
    "openai/gpt-4": "gpt-4",
    "openai/gpt-4-0613": "gpt-4-0613",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09",
    "openai/gpt-4-turbo-preview": "gpt-4-turbo-preview",
    "openai/gpt-4-0125-preview": "gpt-4-0125-preview",
    "openai/gpt-4-1106-preview": "gpt-4-1106-preview",
    # anthropic
    "anthropic/claude-2.1": "claude-2.1",
    "anthropic/claude-3-opus-20240229": "claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "anthropic/claude-3-haiku-20240307": "claude-3-haiku-20240307",
    # mistral
    "mistral/mistral-large-latest": "mistral/mistral-large-latest",
    "mistral/mistral-medium-latest": "mistral/mistral-medium-latest",
    "mistral/mistral-small-latest": "mistral/mistral-small-latest",
    "mistral/codestral-latest": "mistral/codestral-latest",
    "mistral/open-mistral-7b": "mistral/open-mistral-7b",
    "mistral/open-mixtral-8x7b": "mistral/open-mixtral-8x7b",
    "mistral/open-mixtral-8x22b": "mistral/open-mixtral-8x22b",
    # perplexity
    "perplexity/llama-3-sonar-large-32k-online": "perplexity/llama-3-sonar-large-32k-online",
    # cohere
    "cohere/command-r": "cohere_chat/command-r",
    "cohere/command-r-plus": "cohere_chat/command-r-plus",
    # google
    "google/gemini-pro": "gemini/gemini-pro",
    "google/gemini-1.5-pro-latest": "gemini/gemini-1.5-pro-latest",
    "google/gemini-1.5-flash-latest": "gemini/gemini-1.5-flash-latest",
    "google/gemini-1.0-pro-latest": "gemini/gemini-pro",
    # replicate
    "replicate/mistral-7b-instruct-v0.2": "replicate/mistralai/mistral-7b-instruct-v0.2",
    "replicate/mixtral-8x7b-instruct-v0.1": "replicate/mistralai/mixtral-8x7b-instruct-v0.1",
    "replicate/meta-llama-3-70b-instruct": "replicate/meta/meta-llama-3-70b-instruct",
    "replicate/meta-llama-3-8b-instruct": "replicate/meta/meta-llama-3-8b-instruct",
    # togetherai
    "togetherai/Mistral-7B-Instruct-v0.2": "together_ai/mistralai/Mistral-7B-Instruct-v0.2",
    "togetherai/Mixtral-8x7B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "togetherai/Mixtral-8x22B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "togetherai/Phind-CodeLlama-34B-v2": "together_ai/Phind/Phind-CodeLlama-34B-v2",
    "togetherai/Llama-3-70b-chat-hf": "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "togetherai/Llama-3-8b-chat-hf": "together_ai/meta-llama/Llama-3-8b-chat-hf",
    "togetherai/Qwen2-72B-Instruct": "together_ai/Qwen/Qwen2-72B-Instruct",
}


class NotDiamondError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://not-diamond-server.onrender.com/v2/optimizer/router"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class NotDiamondConfig:
    # llm_providers requires at least one provider, setting to gpt-3.5-turbo as default
    llm_providers: Optional[List[Dict[str, str]]] = [
        {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        }
    ]
    tools: Optional[List[Dict[str, str]]] = None
    max_model_depth: int = 1
    # tradeoff params: "cost"/"latency"
    tradeoff: Optional[str] = None
    preference_id: Optional[str] = None
    hash_content: Optional[bool] = False

    def __init__(
        self,
        llm_providers: Optional[List[Dict[str, str]]] = [
            {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            }
        ],
        tools: Optional[str] = None,
        max_model_depth: Optional[int] = 1,
        tradeoff: Optional[str] = None,
        preference_id: Optional[str] = None,
        hash_content: Optional[bool] = False,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
            or k == "llm_providers"
        }


def validate_environment(api_key):
    # oss endpoint doesn't need an api key
    api_key = "" if api_key is None else api_key
    headers = {
        "Authorization": "Bearer " + api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }
    return headers


def get_litellm_model(response: dict) -> str:
    nd_provider = response['providers'][0]['provider']
    nd_model = response['providers'][0]['model']
    nd_provider_model = f"{nd_provider}/{nd_model}"
    litellm_model = notdiamond2litellm[nd_provider_model]
    return litellm_model


def update_litellm_params(litellm_params: dict):
    '''
    Create a new litellm_params dict with non-default litellm_params from the original call, custom_llm_provider and api_base
    '''
    new_litellm_params = dict()
    for k, v in litellm_params.items():
        # all litellm_params have defaults of None or False, except force_timeout
        if (k == "force_timeout" and v != 600) or v:
            new_litellm_params[k] = v
    if "custom_llm_provider" in new_litellm_params: del new_litellm_params["custom_llm_provider"]
    if "api_base" in new_litellm_params: del new_litellm_params["api_base"]
    return new_litellm_params


def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    completion_url = api_base

    ## Load Config
    config = litellm.NotDiamondConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):
            optional_params[k] = v

    data = {
        "messages": messages,
        **optional_params,
    }

    print("optional_params: ", optional_params)
    print("litellm_params: ", litellm_params)

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": completion_url,
        },
    )

    ## MODEL SELECTION CALL
    nd_response = requests.post(
        api_base,
        headers=headers,
        json=data,
    )
    print_verbose(f"raw model_response: {nd_response.text}")

    ## RESPONSE OBJECT
    if nd_response.status_code != 200:
        raise NotDiamondError(
            status_code=nd_response.status_code, message=nd_response.text
        )
    nd_response = nd_response.json()
    litellm_model = get_litellm_model(nd_response)

    ## COMPLETION CALL
    litellm_params = update_litellm_params(litellm_params)

    # Handle Tool Calling
    _is_function_call = True if "tools" in optional_params else False
    if _is_function_call:
        model_response = litellm.completion(
            model=litellm_model,
            messages=messages,
            tools=optional_params["tools"],
            **litellm_params,
        )
    else:
        model_response = litellm.completion(
            model=litellm_model,
            messages=messages,
            **litellm_params,
        )
    return model_response