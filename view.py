from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
import os

model = tf.keras.models.load_model('my_model.keras')

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return jsonify({'message': 'Welcome to my JPN-ANA API!'})


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Check if the request contains a file named 'file'
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file has an allowed extension (e.g., CSV)
        allowed_extensions = {'csv'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format'})

        # Read the CSV file and extract the text data
        df = pd.read_csv(file)
        # Replace 'text_column' with your column name containing text data
        x = df['Japanese']

        # Preprocess the text (similar to what you did during training)
        max_len = 1000
        tok = Tokenizer(num_words=1500)
        tok.fit_on_texts(x)
        sequences = tok.texts_to_sequences(x)
        sequences_matrix = sequence.pad_sequences(
            sequences, maxlen=max_len)

        # Make predictions using the model
        predictions = model.predict(sequences_matrix)
        sentiment_labels = [int(round(max(pred))) for pred in predictions]

        return jsonify({'sentiment_labels': sentiment_labels})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
