import os
import openai
import ell

client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

model_to_use = "gemini-1.5-flash"
# Register the model with your custom client
ell.config.register_model(model_to_use, client)


@ell.simple(model=model_to_use, temperature=0.7)
def prompt_hello():
    return "Hello world tell me a joke"

print(prompt_hello())