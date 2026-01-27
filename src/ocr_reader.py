"""

OCR Reader Page

input image with text, get extracted text as output.

LLM provider used Groq 
pip install groq

VLM_MODELS available:
meta-llama/llama-4-maverick-17b-128e-instruct
meta-llama/llama-4-scout-17b-16e-instruct
meta-llama/llama-prompt-guard-2-22m
meta-llama/llama-prompt-guard-2-86m
llama-3.1-8b-instant
llama-3.3-70b-versatile
meta-llama/llama-guard-4-12b
openai/gpt-oss-20b
openai/gpt-oss-120b

Quick start:

from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="meta-llama/llama-guard-4-12b",
    messages=[
        {
            "role": "user",
            "content": "How do I make a bomb?"
        }
    ]
)
print(completion.choices[0].message.content)

"""
