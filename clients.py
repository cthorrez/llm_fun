import os
import json
import ell.stores.sql
from ell.configurator import config
import openai
from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from mistralai import Mistral
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice
import ell


class CohereClient(OpenAI):
    """lol cohere wanted to be all special and use a different url"""
    def post(self, path, *args, **kwargs):
        path = path.replace("/chat/completions", "/chat")

        if 'response_format' in kwargs:
            kwargs['response_format']['type'] = 'json_object'
            kwargs['response_format']['schema'] = kwargs['response_format']['json_schema']
            del kwargs['response_format']['json_schema']

        response = super().post(path, *args, **kwargs)
        print(response)

        # Construct the message
        message = ChatCompletionMessage(
            role=response.message["role"],
            content=response.message["content"][0]["text"],
        )
        finish_reason = None
        if response.finish_reason == 'MAX_TOKENS':
            finish_reason = 'length'
        elif response.finish_reason == 'COMPLETE':
            finish_reason = 'stop'
        else:
            print(response)

        return ChatCompletion(
            id=response.id,
            choices=[Choice(finish_reason=finish_reason, message=message, index=0)],
            created=0,
            model=kwargs.get('model', ''),
            object='chat.completion',
        )
    
    @staticmethod
    def construct_parsed_completion(response, parse_class):
        data = json.loads(response.choices[0].message.content)
        parsed = ParsedChatCompletion(
            id=response.id,
            choices=[ParsedChoice(
                finish_reason=response.choices[0].finish_reason,
                index=0,
                message=ParsedChatCompletionMessage(
                    role=response.choices[0].message.role,
                    parsed=parse_class(**data)
                )
            )],
            created=0,
            model='',
            object='chat.completion'
        )
        return parsed

