import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras.utils import to_categorical

def build_model(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    return model, history
