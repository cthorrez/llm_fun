import os
import ell.stores
import ell.stores.sql
from ell.configurator import config
import openai
import ell
from cache_utils import CachedOpenAIProvider

def register_clients():

    gemini_client = openai.Client(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    provider = CachedOpenAIProvider('.llm_cache')

    ell.config.register_provider(provider, openai.Client)
    # model = "gemini-1.5-flash"
    model = "gemini-2.0-flash-lite-preview-02-05"
    ell.config.register_model(model, gemini_client)
