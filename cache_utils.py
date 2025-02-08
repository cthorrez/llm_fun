import json
from typing import Literal, Optional, Dict, Any, Callable
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from ell.providers.openai import OpenAIProvider
from pydantic import BaseModel
from diskcache import Cache


class FourChoiceAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D']


def serialize_pydantic(obj):
    """Recursively serialize Pydantic models and handle other types."""
    if isinstance(obj, BaseModel):
        return {
            "__pydantic_model__": obj.__class__.__name__,
            "data": obj.model_dump()
        }
    elif isinstance(obj, dict):
        return {key: serialize_pydantic(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_pydantic(item) for item in obj]
    return obj


class CachedOpenAIProvider(OpenAIProvider):
    dangerous_disable_validation = True

    def __init__(self, cache_directory=".diskcache"):
        super().__init__()
        self.cache = Cache(cache_directory)

    def provider_call_function(self, client: OpenAI, api_call_params: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
        cache_key = self._generate_cache_key(api_call_params)
        
        if cache_key in self.cache:
            def retrieve_from_cache(*args, **kwargs):
                raw_response = self.cache[cache_key]
                # Validate using the correct Pydantic model
                # return ChatCompletion.model_validate_json(raw_response)
                response = json.loads(raw_response)
                response['id'] = 'FUCK'
                return ParsedChatCompletion.model_validate(response)
            return retrieve_from_cache
            
        original_call_function = super().provider_call_function(client, api_call_params)
        
        def cached_call_function(*args, **kwargs):
            response = original_call_function(*args, **kwargs)
            # Store the raw JSON response
            self.cache[cache_key] = response.model_dump_json()
            return response
            
        return cached_call_function

    def _generate_cache_key(self, api_call_params: Optional[Dict[str, Any]]) -> str:
        """Generate consistent cache keys using serialized parameters"""
        if api_call_params is None:
            api_call_params = {}
            
        serialized = serialize_pydantic(api_call_params)
        return json.dumps(serialized, sort_keys=True, default=str)