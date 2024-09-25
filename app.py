from flask import Flask, render_template, request
import numpy as np
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask application
app = Flask(__name__)

# Load the saved LSTM model
model_lstm = load_model('model_lstm.h5')

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define text preprocessing functions
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    # Clean the text
    text = clean_text(text)
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Join words back to text
    text = ' '.join(words)
    return text

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        message = request.form['message']
        # Preprocess the input text
        processed_text = preprocess_text(message)
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([processed_text])
        # Pad the sequence
        max_length = 500  # Use the same max_length as during training
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        # Make prediction
        prediction = model_lstm.predict(padded_sequence)
        # Interpret the prediction
        if prediction >= 0.5:
            result = 'Real News'
        else:
            result = 'Fake News'
        # Render the result template with the prediction
        return render_template('result.html', prediction=result)
    else:
        return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
