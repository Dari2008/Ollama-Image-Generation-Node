# import ollama as ai
# client = ai.Client("http://192.168.178.138:11434")

# response = client.generate(model="x/flux2-klein", prompt="A cute cat")

# print(response)

import base64
from openai import OpenAI

client = OpenAI(
    base_url='http://192.168.178.138:11434/v1/',
    api_key='ollama',  # required but ignored
)

response = client.images.generate(
    model='x/flux2-klein',
    prompt='A cute cat',
    size='1024x1024',
    response_format='b64_json',
)

# Save the image
image_data = base64.b64decode(response.data[0].b64_json)
with open('output.png', 'wb') as f:
    f.write(image_data)

print("Image saved to output.png")