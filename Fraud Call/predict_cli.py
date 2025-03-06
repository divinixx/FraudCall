from utils.model import load_model
from utils.predictor import predict_fraud

def fraud_call_detection_cli():
    # Load the model
    model_path = "D:\\Python\\AIML\\Fraud Call\\models\\fraud_model.pkl"
    vectorizer_path = "D:\\Python\\AIML\\Fraud Call\\models\\tfidf_vectorizer.pkl"
    
    try:
        model, vectorizer = load_model(model_path, vectorizer_path)
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return
    
    print("===== Fraud Call Detection System =====")
    print("Enter 'quit' to exit the system")
    
    while True:
        user_input = input("\nEnter call transcript to analyze: ")
        if user_input.lower() == 'quit':
            break
            
        prediction, probability = predict_fraud(user_input, model, vectorizer)
        
        if prediction == 1:
            print(f"⚠️ FRAUD ALERT: This call is likely fraudulent (Confidence: {probability:.2%})")
        else:
            print(f"✅ LEGITIMATE: This call appears to be legitimate (Confidence: {(1-probability):.2%})")

if __name__ == "__main__":
    fraud_call_detection_cli()