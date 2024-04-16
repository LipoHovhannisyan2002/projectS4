import spacy
import speech_recognition as sr

# Load SpaCy with a custom Armenian language model (if available)
# nlp = spacy.load("custom_armenian_model")

# If a custom model isn't available, use SpaCy's default model
nlp = spacy.blank("hy")

# Example Armenian text
armenian_text = "Բարև, այս տեքստը ներքին լեզվով է։"

# Tokenization
doc = nlp(armenian_text)
tokens = [token.text for token in doc]
print("Tokens:", tokens)

# Speech Recognition
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak something in Armenian:")
    audio = recognizer.listen(source)

try:
    recognized_text = recognizer.recognize_google(audio, language="hy-AM")
    print("Recognized Speech (Armenian):", recognized_text)
except sr.UnknownValueError:
    print("Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
