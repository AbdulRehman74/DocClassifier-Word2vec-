from flask import Flask, request, jsonify
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Load the saved classifier, Word2Vec model, and label encoder
model_data = joblib.load('text_classifier_model_with_w2v.joblib')

classifier = model_data['classifier']
w2v_model = model_data['w2v_model']
label_encoder = model_data['label_encoder']

# Other necessary preprocessing and objects should be loaded here

def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def vectorize(sentence):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(200)  # Adjust to match the vector size used during training
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    document = request.json['document']
    input_vector = vectorize(preprocess(document))
    input_vector = input_vector.reshape(1, -1)
    predicted_label = classifier.predict(input_vector)
    
    # Inverse transform the label to get the predicted category
    predicted_category = label_encoder.inverse_transform(predicted_label)
    
    return jsonify({'predicted_category': predicted_category[0]})

app.run(port=5000)
