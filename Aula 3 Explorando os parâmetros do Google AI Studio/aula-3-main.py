"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""
from pathlib import Path
import hashlib
import google.generativeai as genai

genai.configure(api_key="SUA_API_KEY")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: Dada a imagem de um animal, escreva uma descrição de marketing envolvente para uma loja de óculos",
    genai.upload_file("/duck-funny.jpg"),
  "output: Você está pronto para ver o pato mais engraçado que você já viu?",
]

response = model.generate_content(prompt_parts)
print(response.text)