import whisper

model = whisper.load_model("tiny")

result = model.transcribe(
    "data/test_audio.wav",
    language="en",
    fp16=False
)

print("\n========================")
print("TRANSCRIPTION")
print("========================")
print(result["text"].strip())