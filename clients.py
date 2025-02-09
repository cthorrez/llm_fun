import os
import ell.stores
import ell.stores.sql
from ell.configurator import config
import openai
from mistralai import Mistral
import ell
from cache_utils import CachedOpenAIProvider

def register_clients(timeout=10.0):
    provider = CachedOpenAIProvider('.llm_cache', timeout=timeout)
    ell.config.register_provider(provider, openai.Client)

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
    ]
    for gemini_model in gemini_models:
        ell.config.register_model(gemini_model, gemini_client)

    mistral_client = openai.Client(
        base_url="https://api.mistral.ai/v1",
        api_key=os.environ["MISTRAL_API_KEY"],
    )
    mistral_models = [
        "mistral-small-latest",
        "open-mistral-nemo"
    ]
    for mistral_model in mistral_models:
        ell.config.register_model(mistral_model, mistral_client)
