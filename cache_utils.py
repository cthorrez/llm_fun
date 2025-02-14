import time
import json
from typing import Literal, Optional, Dict, Any, Callable
from openai import OpenAI
from google import genai
from google.genai import types
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice
from ell.providers.openai import OpenAIProvider
from pydantic import BaseModel
from diskcache import Cache
from clients import CohereClient


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

    def __init__(self, cache_directory=".diskcache", timeout=10.0):
        super().__init__()
        self.cache = Cache(cache_directory)
        self.timeout = timeout
        self.last_call_time = time.time() - timeout

    def provider_call_function(self, client: OpenAI, api_call_params: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
        force_retry = api_call_params.get('force_retry')
        if force_retry is not None:
            del api_call_params['force_retry']
        assert 'force_retry' not in api_call_params, "?????"
        is_structured = api_call_params.get('response_format')

        parse_class = api_call_params['response_format']

        if isinstance(client, CohereClient) and is_structured:
            api_call_params['response_format'] = { 
                "type": "json_object",
                "schema": api_call_params['response_format'].model_json_schema()
            }
            
        if not is_structured:
            api_call_params['stream'] = False
        cache_key = self._generate_cache_key(api_call_params)
        
        if (cache_key in self.cache) and (not force_retry):
            def retrieve_from_cache(*args, **kwargs):
                raw_response = self.cache[cache_key]
                return_val = raw_response
                if is_structured:
                    response = json.loads(raw_response)
                    if isinstance(client, CohereClient):
                        cc = ChatCompletion.model_validate(response)
                        return_val = client.construct_parsed_completion(cc, parse_class)

                    else:
                        response['id'] = '' # workaround, gemini doesn't add an id
                        return_val = ParsedChatCompletion[parse_class].model_validate(response)
                return return_val
            return retrieve_from_cache
            
        original_call_function = super().provider_call_function(client, api_call_params)
        
        def call_function_and_store_to_cache(*args, **kwargs):
            wait_time = max((self.timeout - (time.time() - self.last_call_time)),0)
            if wait_time > 0:
                print(f'Waiting {wait_time:.4f} seconds before making next call')
            time.sleep(wait_time)
            if 'stream_options' in kwargs:
                del kwargs['stream_options']
            
            # workaround for mistral
            if kwargs['messages'][-1]['role'] == 'assistant':
                kwargs['messages'][-1]['prefix'] = True

            response = original_call_function(*args, **kwargs)
            self.last_call_time = time.time()
            cache_val = response
            if is_structured:
                cache_val = response.model_dump_json()

            # don't cache content filtered requests  
            if response.choices[0].finish_reason != 'content_filter':
                self.cache[cache_key] = cache_val

            if isinstance(client, CohereClient) and is_structured:
                parsed = client.construct_parsed_completion(response, parse_class)
                return parsed

            return response
            
        return call_function_and_store_to_cache

    def _generate_cache_key(self, api_call_params: Optional[Dict[str, Any]]) -> str:
        """Generate consistent cache keys using serialized parameters"""
        if api_call_params is None:
            api_call_params = {}
            
        serialized = serialize_pydantic(api_call_params)
        return json.dumps(serialized, sort_keys=True, default=str)