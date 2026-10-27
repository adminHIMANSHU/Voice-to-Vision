import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading VLM model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

model.eval()

print("VLM model loaded successfully.")

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    cv2.imshow("VoiceToVision - VLM Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # SPACE
        image_path = "data/vlm_test.jpg"
        cv2.imwrite(image_path, frame)

        print(f"\nImage saved: {image_path}")
        print("Analyzing image with VLM...")

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": "Describe what is visible in this image. Identify important objects and explain the scene clearly for a visually impaired person."
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

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100
            )

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print("\n========== VLM RESULT ==========")
        print(generated_text)
        print("================================\n")

camera.release()
cv2.destroyAllWindows()