import types
import json
import requests
from typing import Callable, Optional, Dict, List
import httpx

import litellm
from litellm.utils import ModelResponse, get_api_base


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

# dict to map notdiamond providers and models to litellm providers and models
notdiamond2litellm = {
    "cohere/command-r": "cohere_chat/command-r",
    "cohere/command-r-plus": "cohere_chat/command-r-plus",
    "google/gemini-pro": "gemini/gemini-pro",
    "google/gemini-1.5-pro-latest": "gemini/gemini-1.5-pro-latest",
    "google/gemini-1.5-flash-latest": "gemini/gemini-1.5-flash-latest",
    "google/gemini-1.0-pro-latest": "gemini/gemini-pro",
    "replicate/mistral-7b-instruct-v0.2": "replicate/mistralai/mistral-7b-instruct-v0.2",
    "replicate/mixtral-8x7b-instruct-v0.1": "replicate/mistralai/mixtral-8x7b-instruct-v0.1",
    "replicate/meta-llama-3-70b-instruct": "replicate/meta/meta-llama-3-70b-instruct",
    "replicate/meta-llama-3-8b-instruct": "replicate/meta/meta-llama-3-8b-instruct",
    "togetherai/Mistral-7B-Instruct-v0.2": "together_ai/mistralai/Mistral-7B-Instruct-v0.2",
    "togetherai/Mixtral-8x7B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "togetherai/Mixtral-8x22B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "togetherai/Phind-CodeLlama-34B-v2": "together_ai/Phind/Phind-CodeLlama-34B-v2",
    "togetherai/Llama-3-70b-chat-hf": "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "togetherai/Llama-3-8b-chat-hf": "together_ai/meta-llama/Llama-3-8b-chat-hf",
    "togetherai/Qwen2-72B-Instruct": "together_ai/Qwen/Qwen2-72B-Instruct",
}

def get_litellm_model_provider(response: dict) -> str:
    nd_provider = response['providers'][0]['provider']
    nd_model = response['providers'][0]['model']
    provider_model = f"{nd_provider}/{nd_model}"
    if provider_model in notdiamond2litellm:
        provider_model = notdiamond2litellm[provider_model]
    return provider_model


def update_litellm_params(litellm_params: dict, litellm_provider_model: str):
    '''
    Update litellm_params after model selection, otherwise `custom_llm_provider=notdiamond` leading to infinite loop into nd.completion()
    '''
    custom_llm_provider = litellm_provider_model.split("/")[0]
    litellm_params["custom_llm_provider"] = custom_llm_provider
    litellm_params["api_base"] = None
    # first completion call adds three extra params that are not in list of litellm_params in main.completion()c
    if "model_alias_map" in litellm_params: del litellm_params["model_alias_map"]
    if "completion_call_id" in litellm_params: del litellm_params["completion_call_id"]
    if "stream_response" in litellm_params: del litellm_params["stream_response"]
    return litellm_params


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

    ## Handle Tool Calling
    # if "tools" in optional_params:
    #     _is_function_call = True
    #     tool_calling_system_prompt = construct_cohere_tool(
    #         tools=optional_params["tools"]
    #     )
    #     optional_params["tools"] = tool_calling_system_prompt
    
    data = {
        "messages": messages,
        **optional_params,
    }

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
    litellm_provider_model = get_litellm_model_provider(nd_response)
    litellm_provider, litellm_model = litellm_provider_model.split("/")

    ## COMPLETION CALL
    # using litellm_params with completion() call leads to an error
    litellm_params = update_litellm_params(litellm_params, litellm_provider_model)

    model_response = litellm.completion(
        model=litellm_model,
        messages=messages,
        **litellm_params,
    )
    return model_response