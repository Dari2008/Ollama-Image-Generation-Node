import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.178.138:11434/v1/",
    api_key="ollama",
)

# ---- 1. STREAM PROMPT CREATION ----
print("Generating prompt:\n")

stream = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "Create a detailed prompt for a cute pixel art cat"}
    ],
    stream=True,
)

prompt = ""

for chunk in stream:
    if chunk.choices[0].delta.content:
        text = chunk.choices[0].delta.content
        prompt += text
        print(text, end="", flush=True)

print("\n\nFinal prompt:\n", prompt)


# ---- 2. IMAGE GENERATION (NON-STREAMING) ----
print("\nGenerating image...")

response = client.images.generate(
    model="x/flux2-klein",
    prompt=prompt,
    size="1024x1024",
)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Image saved to output.png")