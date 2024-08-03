import pyttsx3
engine = pyttsx3.init()
engine.setProperty("rate", 128)
engine.say("I will speak this text")
for voice in engine.getProperty('voices'):
    print(voice)
    print("test")
engine.runAndWait()