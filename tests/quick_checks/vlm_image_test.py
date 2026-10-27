import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_PATH = "data/vlm_test.jpg"

print("Loading VLM...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float32
)

model.eval()

print("VLM loaded successfully.")

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Describe this scene for a visually impaired person. "
                    "Identify the important objects, their approximate positions, "
                    "and anything that may require attention."
                )
            }
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
)

print("Analyzing image...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100
    )

result = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n========== VLM RESULT ==========")
print(result)
print("================================")