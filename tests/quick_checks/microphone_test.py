import sounddevice as sd
import soundfile as sf
import numpy as np

DEVICE = 9
DURATION = 5

device_info = sd.query_devices(DEVICE)
SAMPLE_RATE = int(device_info["default_samplerate"])

print(f"Microphone : {device_info['name']}")
print(f"Sample Rate: {SAMPLE_RATE}")
print("Recording... Speak now!")

recording = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    recording.append(indata.copy())

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE,
    callback=callback,
):
    sd.sleep(DURATION * 1000)

audio = np.concatenate(recording, axis=0)

rms = np.sqrt(np.mean(audio**2))
print(f"Audio Level (RMS): {rms:.4f}")

sf.write("data/test_audio.wav", audio, SAMPLE_RATE)

print("Saved -> data/test_audio.wav")