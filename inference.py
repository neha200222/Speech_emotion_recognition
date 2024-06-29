from keras.utils import pad_sequences

def predict_emotion(model, tokenizer, text):
    new_text_seq = tokenizer.texts_to_sequences([text])
    new_text_padded = pad_sequences(new_text_seq, padding='post', maxlen=35)
    predictions = model.predict(new_text_padded)
    predicted_class_index = predictions.argmax(axis=-1)
    if predicted_class_index[0] == 0:
        return "Positive Sentiment"
    elif predicted_class_index[0] == 1:
        return "Negative Sentiment"
    else:
        return "Neutral Sentiment"
