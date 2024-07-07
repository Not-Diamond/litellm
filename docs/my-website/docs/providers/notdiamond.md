# Not Diamond
[Not Diamond](https://www.notdiamond.ai/) automatically determines which model is best-suited to respond to any query, for improved quality, and reduced cost and latency. LiteLLM supports automatic model routing to all [models supported by Not Diamond](https://notdiamond.readme.io/v0.1.0-beta/docs/supported-models).

## Pre-Requisites
`pip install litellm`

## Required API Keys
Follow this [link](https://notdiamond.readme.io/v0.1.0-beta/docs/api-keys) to create your Not Diamond API key. Additionally, provide API keys for all providers that you want to route between.

```python
os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"  # NOTDIAMOND_API_KEY
# provide API keys for providers
```

:::info

Not Diamond API fails requests when `llm_providers` are not passed. Please provide your desired LLM providers and models to route between.
:::

## Usage

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "Hey! how's it going?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers
)
print(response)
```

## Usage - Streaming
Set `stream=True` when calling completion to stream responses.

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "Hey! how's it going?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers,
  stream=True
)
for chunk in response:
    print(chunk)
```

## Usage - Function Calling
Function calling is also supported through the `tools` parameter for [models that support function calling](https://notdiamond.readme.io/v0.1.0-beta/docs/function-calling).

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "What is 2 + 5?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

tools = [
    {
        "type": "function",
        "function": {
        "name": "add",
        "description": "Adds a and b.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        },
    },
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers,
  tools=tools
)
print(response)
```