import os
import ell.stores
import ell.stores.sql
from ell.configurator import config
import openai
from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from mistralai import Mistral
import ell
from cache_utils import CachedOpenAIProvider
from clients import CohereClient



def register_clients(timeout=10.0):
    provider = CachedOpenAIProvider('.llm_cache', timeout=timeout)
    ell.config.register_provider(provider, openai.Client)
    ell.config.register_provider(provider, CohereClient)

    gemini_client = openai.Client(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    gemini_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05",
    ]
    for gemini_model in gemini_models:
        ell.config.register_model(gemini_model, gemini_client)

    mistral_client = openai.Client(
        base_url="https://api.mistral.ai/v1",
        api_key=os.environ["MISTRAL_API_KEY"],
    )
    mistral_models = [
        "mistral-small-latest",
        "open-mistral-nemo",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
    ]
    for mistral_model in mistral_models:
        ell.config.register_model(mistral_model, mistral_client)

    scaleway_client = openai.Client(
        base_url = "https://api.scaleway.ai/8b2c7bde-831d-4972-b96f-c03d47763941/v1",
        api_key = os.environ["SCALEWAY_API_KEY"]
    )
    scaleway_models = ["llama-3.3-70b-instruct", "deepseek-r1-distill-llama-70b", "deepseek-r1"]
    for scaleway_model in scaleway_models:
        ell.config.register_model(scaleway_model, scaleway_client)


    cohere_client = CohereClient(
        base_url = "https://api.cohere.com/v2",
        api_key = os.environ["COHERE_API_KEY"]
    )
    cohere_models = ["command-r-plus-08-2024"]
    for cohere_model in cohere_models:
        ell.config.register_model(cohere_model, cohere_client)
