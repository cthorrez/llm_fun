from ell.providers.openai import OpenAIProvider
from openai import OpenAI
from diskcache import Cache
import json
from typing import Optional, Dict, Any, Callable
from pydantic import BaseModel

class CachedOpenAIProvider(OpenAIProvider):
    dangerous_disable_validation = True

    def __init__(self, cache_directory=".diskcache"):
        super().__init__()
        self.cache = Cache(cache_directory)  # Initialize diskcache

    def provider_call_function(self, client: OpenAI, api_call_params: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
        # Generate a unique cache key based on the API call parameters
        api_call_params['stream'] = False
        cache_key = self._generate_cache_key(api_call_params)

        # Check if the response is already in the cache
        if cache_key in self.cache:
            print("Returning cached response")
            def get(*args, **kwargs):
                return self.cache[cache_key]
            return get
        # If not in cache, get the original call function
        original_call_function = super().provider_call_function(client, api_call_params)

        # Wrap the original call function to cache the response
        def cached_call_function(*args, **kwargs):
            response = original_call_function(*args, **kwargs)
            # print(response)
            self.cache[cache_key] = response  # Cache the response
            print("Caching response")
            return response

        return cached_call_function

    def _generate_cache_key(self, api_call_params: Optional[Dict[str, Any]]) -> str:
        """
        Generate a unique cache key based on the API call parameters.
        """
        if api_call_params is None:
            api_call_params = {}
        # Convert the parameters to a JSON-serializable string
        return json.dumps(api_call_params, sort_keys=True)  # Ensure consistent ordering