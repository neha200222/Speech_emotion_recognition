import speech_recognition as sr
from inference import predict_emotion
from preprocess import load_and_preprocess_data
from train import build_model

def record_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    return audio

def convert_audio_to_text(audio):
    recognizer = sr.Recognizer()

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Load the trained model and tokenizer
x_train, y_train, x_test, y_test, tokenizer = load_and_preprocess_data('data/train.csv', 'data/validation.csv')
model = build_model(input_dim=20000, output_dim=5, input_length=35)
model.load_weights('models/emotion_model.h5')

# Record audio from the microphone
audio_data = record_audio()

# Convert audio to text
text_result = convert_audio_to_text(audio_data)

# Predict the emotion of the text
emotion = predict_emotion(model, tokenizer, text_result)

# Display the emotion
print("Predicted Emotion:", emotion)
