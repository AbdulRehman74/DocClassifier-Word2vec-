import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib

with open('docs.json', 'r') as json_file:
    data = json.load(json_file)

categories = [item['category'] for item in data]
text_data = [item['document'] for item in data]

text_data = [text.lower() for text in text_data]

label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

X_train, X_test, y_train, y_test = train_test_split(text_data, encoded_categories, test_size=0.3, random_state=42)

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


X_train = [preprocess(text) for text in X_train]
X_test = [preprocess(text) for text in X_test]


sentences = [sentence.split() for sentence in X_train]
w2v_model = Word2Vec(sentences, vector_size=200, window=10, min_count=5, workers=4)
# Vectorize the text data
def vectorize(sentence):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

X_train = np.array([vectorize(sentence) for sentence in X_train])
X_test = np.array([vectorize(sentence) for sentence in X_test])

classifier = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, F1 score, and support for each class
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(classification_rep)

def predict_category(document, classifier, label_encoder):
    input_vector = vectorize(document)
    input_vector = input_vector.reshape(1, -1)
    predicted_label = classifier.predict(input_vector)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]

model_data = {
    'classifier': classifier,
    'w2v_model': w2v_model,
    'label_encoder': label_encoder,
}

joblib.dump(model_data, 'text_classifier_model_with_w2v.joblib')

while True:
    input_document = input("Enter a document (paste or type) and press Enter when done. Type 'exit' to finish: ")
    input_document = input_document.lower()
    if input_document.strip().lower() == "exit":
        break

    predicted_category = predict_category(input_document, classifier, label_encoder)
    print(f"Predicted Category: {predicted_category}")
