from data_preprocessing import *
import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('saved_models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
l_le = joblib.load('saved_models/le.pkl')
l_model = tf.keras.models.load_model('saved_models/transformer.keras')

df = pd.read_excel(open('Consolidated.xlsx', 'rb'), sheet_name='Consolidated')

print(len(df['Review'].values[1:50]))

def predict_for_custom_review(text):
    X = preprocess_text(text)
    X_train_sequences = tokenizer.texts_to_sequences([X])
    max_seq_length = 250
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    y_pred = np.argmax(l_model.predict(X_train_padded), axis=1) 
    return l_le.inverse_transform(y_pred)[0]

interface = gr.Interface(fn=predict_for_custom_review, 
                         inputs='text', 
                         outputs="label",
                         examples=df[['Review','Target']].sample(n=50).to_numpy().tolist(),
                         title="Fake Review Detector using transformer")
interface.launch()