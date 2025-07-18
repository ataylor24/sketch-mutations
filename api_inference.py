import copy
from tqdm import tqdm
import os
import requests
from openai import OpenAI, AzureOpenAI
from utils import format_prompt

def get_api_key(api_type):
    """Get API key from environment variables based on API type."""
    if api_type == "azure":
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        return key
    elif api_type == "inception":
        key = os.getenv("INCEPTION_API_KEY")
        if not key:
            raise ValueError("INCEPTION_API_KEY not found in environment variables")
        return key
    elif api_type == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        return key
    else:  # openai
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return key

def load_inference_client(config):
    """
    Load the appropriate client for API-based inference.
    """
    llm_name = config["model_name"]
    api_type = config.get("api_type", "openai")
    
    # Get API key first to fail fast if missing
    api_key = get_api_key(api_type)
    
    if api_type == "azure":
        print(f'Using Azure OpenAI: {llm_name}')
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
            timeout=200
        )
    elif api_type == "inception":
        print(f'Using Inception Labs client (inceptionlabs-python interface): {llm_name}')
        class InceptionClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.base_url = 'https://api.inceptionlabs.ai/v1'
                self.headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
            
            def chat_completions_create(self, model, messages, max_tokens=None):
                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens
                    }
                )
                return response.json()
        
        client = InceptionClient(api_key=api_key)
    elif api_type == "deepseek":
        print(f'Using DeepSeek client (OpenAI-python interface): {llm_name}')
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=1000)
    else:
        print(f'Using OpenAI client (OpenAI-python interface): {llm_name}')
        client = OpenAI(api_key=api_key, timeout=1000)
        
    return client

def _inference(client, prompt_msg, config) -> str:
    """
    Send <messages> to the provider and return the assistant’s reply text.
    kwargs are forwarded (e.g. max_tokens).
    """
    api_type = config["api_type"]
    model_name = config["model_name"]
    max_tokens = config["max_tokens"]
    
    kwargs = {
        "instructions": "You are a helpful assistant."
        }
    
    if api_type == "inception":
        return NotImplementedError("Inception is not supported for inference")
        # resp = client.chat_completions_create(model=model_name, messages=messages, **kwargs)
        # return resp["choices"][0]["message"]["content"]
    else:  # openai, deepseek, azure → same OpenAI-python interface but with chat completion endpoint

        messages = [
            {"role": "system", "content": kwargs.pop("instructions")},
            {"role": "user", "content": prompt_msg},
        ]

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        return resp
