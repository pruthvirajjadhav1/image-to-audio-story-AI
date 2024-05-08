from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from openai import OpenAI
import os
import requests

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

def image2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']

    to_story(text)


def to_story(discription):
    client = OpenAI()
    completion = client.chat.completions.create(
        model= "gpt-3.5-turbo",
        messages = [
            {"role":"system", "content": "You are an assistant which creates story from just a description of the scenerio"},
            {"role":"user", "content": discription}
        ]
    )
    story_content = completion.choices[0].message.content
    text2speech(story_content)



def text2speech(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payloads = {
        "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)



image2text('img.jpg')
