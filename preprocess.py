import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def load_and_preprocess_data(train_file_path, validation_file_path):
    train_ds = pd.read_csv(train_file_path, encoding='latin1')
    validation_ds = pd.read_csv(validation_file_path, encoding='latin1')
    
    train_ds.fillna('', inplace=True)
    validation_ds.fillna('', inplace=True)
    
    def func(sentiment):
        if sentiment == 'positive':
            return 0
        elif sentiment == 'negative':
            return 1
        else:
            return 2
            
    train_ds['sentiment'] = train_ds['sentiment'].apply(func)
    validation_ds['sentiment'] = validation_ds['sentiment'].apply(func)
    
    x_train = train_ds['text'].tolist()
    y_train = train_ds['sentiment'].tolist()
    x_test = validation_ds['text'].tolist()
    y_test = validation_ds['sentiment'].tolist()
    
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    
    x_train = pad_sequences(x_train, padding='post', maxlen=35)
    x_test = pad_sequences(x_test, padding='post', maxlen=35)
    
    return x_train, y_train, x_test, y_test, tokenizer
