import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from models.saved_models.datapreprocessing import *
from keras import saving
import pandas as pd
import pickle
import numpy as np



from xgboost import XGBClassifier, cv, DMatrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import *
import numpy as np
import pandas as pd

import tensorflow as tf
import joblib, json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, RepeatVector, Input, Reshape, Flatten
from fastapi.middleware.cors import CORSMiddleware





app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)



@app.get('/')
def index():
    return {'message': 'Fake Review Predictor ML API'}


@saving.register_keras_serializable()
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
        Dense(1024, input_dim=shape[1], activation='relu'),
        Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
        Dense(128, input_dim=latent_dim, activation='relu'),
        Dense(shape[1], activation='relu'),
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Review(BaseModel):
  reviewText: str

@app.post("/detectFakeReview")
def predictXgBoost(data:Review):
  print("Input Received:")
  print(predict_for_custom_review(data.reviewText))
  return predict_for_custom_review(data.reviewText)

xgb_model_loaded = pickle.load(open("models/saved_models/xgb_reg.pkl", "rb"))
l_vectorizer = joblib.load('models/saved_models/vectorizer.pkl')
custom_objects = {'Autoencoder': Autoencoder}
l_autoencoder = tf.keras.models.load_model('models/saved_models/autoencoder.keras', custom_objects=custom_objects)
l_le=pickle.load(open("models/saved_models/le.pkl", "rb"))


def predict_for_custom_review(text):
    ff = preprocess_text(text)
    hh = l_vectorizer.transform([ff]).toarray()
    df_encoded = pd.DataFrame(l_autoencoder.encoder(hh).numpy())
    y_pred = xgb_model_loaded.predict(df_encoded)
    print(y_pred[0])
    rval = ""
    if y_pred[0] == 0:
        rval = "False Negative"
    elif y_pred[0] == 1:
        rval = "False Positive"
    elif y_pred[0] == 2:
        rval = "True Negative"
    elif y_pred[0] == 3:
        rval = "True Positive"

    return rval

def read_key_from_json_list(json_data, key):
    # Parse JSON data
    data = json.loads(json_data)
    
    # Extract the specified key from each item in the list
    extracted_values = [item[key] for item in data]
    
    return extracted_values

@app.get("/reviewlist")    
def excel_to_json():
    # Read data from Excel file
    excel_file_path = "models/demo.xlsx"
    df = pd.read_excel(excel_file_path)
    # Convert DataFrame to JSON
    # json_data = df_final_test[["Review","Target"]].to_json(orient="records")
    # Convert DataFrame to array of JSON objects
    json_data = df.to_dict(orient='records')

    return json_data



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000 )