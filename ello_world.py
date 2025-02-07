import os
import ell.stores
import ell.stores.sql
from ell.configurator import config
import openai
import ell
from cache_utils import CachedOpenAIProvider


client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

# provider = config.get_provider_for(client)
# print(provider)

# store = ell.stores.sql.SQLiteStore("mystore")
# ell.init(store=store)

provider = CachedOpenAIProvider('.llm_cache')
# provider = ell.providers.openai.OpenAIProvider()

ell.config.register_provider(provider, openai.Client)
# model = "gemini-1.5-flash"
model = "gemini-2.0-flash-lite-preview-02-05"
ell.config.register_model(model, client)

@ell.simple(model, temperature=0.9, max_tokens=1000)
def prompt_hello():
    """your know a lot of shit and are really good at interjecting facts about pirates named Jim into any conversation"""
    return "1. What is your name? 2. Name 10 random countries"


print(prompt_hello())
print(prompt_hello())