from data_preprocessing import *
import gradio as gr
import pandas as pd
import numpy as np

l_le, l_vectorizer, l_autoencoder, l_XGB_clf = load_models()

df = pd.read_excel(open('Consolidated.xlsx', 'rb'), sheet_name='Consolidated')

print(len(df['Review'].values[1:50]))

def predict_for_custom_review(text):
    ff = preprocess_text(text)
    hh = l_vectorizer.transform([ff]).toarray()
    df_encoded = pd.DataFrame(l_autoencoder.encoder(hh).numpy())
    y_pred = l_XGB_clf.predict(df_encoded)
    return l_le.inverse_transform(y_pred)[0]

text = """I will say the hotel was nice and the staff was friendly. The reason for my review is hidden fees. I thought I had paid for my hotel fully through booking online. Two days later I saw an additional charge on my credit card. It was for a “destination fee”. I was not informed of this ahead of time and find this to be dishonest."""

interface = gr.Interface(fn=predict_for_custom_review, 
                         inputs='text', 
                         outputs="label",
                         examples=df[['Review','Target']].sample(n=50).to_numpy().tolist(),
                         title="Fake Review Detector using encoder and xgboost")
interface.launch()