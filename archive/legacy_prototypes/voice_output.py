import pyttsx3


engine = pyttsx3.init()

engine.setProperty("rate", 155)
engine.setProperty("volume", 1.0)


def speak(message):

    print(f"VOICE: {message}")

    engine.say(message)
    engine.runAndWait()


if __name__ == "__main__":

    print("VoiceToVision Voice Output Test")
    print("-" * 50)

    speak("Stop. Person ahead.")

    print("-" * 50)
    print("Voice test completed.")