import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, RepeatVector, Input, Reshape, Flatten
from tensorflow.keras.models import *
import joblib
from xgboost import XGBClassifier

def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()

    # 2. Removing special characters & punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 

    # 3. Tokenization
    tokens = nltk.word_tokenize(text)

    # 4. Stopword Removal 
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # 5. Lemmatization (Change words to root form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # 6. Joining tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def save_models(le, vectorizer, autoencoder, XGB_clf):
    joblib.dump(le, 'saved_models/le.pkl')
    joblib.dump(vectorizer, 'saved_models/vectorizer.pkl')
    autoencoder.save('saved_models/autoencoder.keras')
    XGB_clf.save_model('saved_models/xgboost.json')


def load_models():
    l_le = joblib.load('saved_models/le.pkl')
    l_vectorizer = joblib.load('saved_models/vectorizer.pkl')
    custom_objects = {'Autoencoder': Autoencoder}
    l_autoencoder = tf.keras.models.load_model('saved_models/autoencoder.keras', custom_objects=custom_objects)
    l_XGB_clf = XGBClassifier(objective='multi:softmax')
    l_XGB_clf.load_model("saved_models/xgboost.json")
    return l_le, l_vectorizer, l_autoencoder, l_XGB_clf

@tf.keras.saving.register_keras_serializable()
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