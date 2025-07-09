import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import pandas


def get_llm_client(llm_choice):
    if llm_choice == "GROQ":
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        return client
    else:
        raise ValueError("Invalid LLM choice. Please choose 'GROQ'.")

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_CHOICE = "GROQ"

print(f"GROQ_API_KEY exists and begins {GROQ_API_KEY[:2]}...")
client = get_llm_client(LLM_CHOICE) #load the client from above function
MODEL = "llama-3.3-70b-versatile" #GROQ's model

print(f"LLM_CHOICE: {LLM_CHOICE} - MODEL: {MODEL}")

with open('system_message.txt', 'r') as file:
    system_message = ' '.join(line.strip() for line in file.readlines())

def chat(message, history):
    messages = (
        [{"role": "system", "content": system_message}]
        + [{"role": "user", "content": message}]
    )
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True, temperature=0.0
    )

    # Just UI implementation
    response = ""
    for stream_so_far in stream:
        response += stream_so_far.choices[0].delta.content or ""
        yield response
USERNAME = os.getenv("GRADIO_USERNAME")
PASSWORD = os.getenv("GRADIO_PASSWORD")
demo = gr.ChatInterface(fn=chat, type="messages")
demo.launch(share = True, auth=(USERNAME, PASSWORD))