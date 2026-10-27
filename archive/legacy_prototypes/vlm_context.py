import cv2
import torch

from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


print("Loading VLM model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

print("VLM model loaded successfully.")


def analyze_scene(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prompt = (
        "Describe this scene for a visually impaired person. "
        "Focus on important objects, obstacles, people, doors, stairs, "
        "vehicles and anything that may affect safe movement. "
        "Keep the description short and practical."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100
        )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return result


print("\nVoiceToVision - VLM CONTEXT TEST")
print("-" * 50)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit


print("Camera started.")
print("Press SPACE to analyze the current scene.")
print("Press Q to quit.")
print("-" * 50)


try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame could not be read.")
            continue

        cv2.imshow(
            "VoiceToVision - VLM Context",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            print("\nAnalyzing scene with SmolVLM...")
            print("Please wait...")

            result = analyze_scene(frame)

            print("\nVLM CONTEXT")
            print("-" * 50)
            print(result)
            print("-" * 50)

        elif key == ord("q"):
            print("\nQuit requested.")
            break


except KeyboardInterrupt:

    print("\nProgram interrupted by user.")


finally:

    camera.release()
    cv2.destroyAllWindows()

    print("VLM Context Test stopped.")