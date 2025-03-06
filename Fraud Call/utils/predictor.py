from utils.preprocessing import preprocess_text

def predict_fraud(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][1]
    
    return prediction, probability