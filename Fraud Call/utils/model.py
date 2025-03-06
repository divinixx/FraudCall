import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

def train_model(corpus, labels, max_features=500, test_size=0.20, random_state=0):
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = MultinomialNB().fit(X_train, y_train)
    print("Model has been trained.")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print evaluation metrics
    print("Confusion matrix:\n", conf_matrix)
    print("Accuracy score:", acc)
    print("Recall score:", recall)
    print("Classification report:\n", report)
    
    return model, vectorizer

def save_model(model, vectorizer, model_path, vectorizer_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def load_model(model_path, vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer