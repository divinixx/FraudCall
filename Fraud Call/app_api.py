import os
import pickle
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Call Detection API",
    description="API for detecting fraudulent calls based on transcript text from India",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define request model
class CallTranscript(BaseModel):
    text: str

# Define response model
class PredictionResult(BaseModel):
    is_fraud: bool
    confidence: float
    message: str

# Load model and vectorizer
MODEL_PATH = "D:\\Python\\AIML\\Fraud Call\\models\\fraud_model.pkl"
VECTORIZER_PATH = "D:\\Python\\AIML\\Fraud Call\\models\\tfidf_vectorizer.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None

# Enhanced preprocess text function for Indian context with Hindi
def preprocess_text(text):
    ps = WordNetLemmatizer()
    
    # List of common Hindi words that might appear in transliterated text
    hindi_stopwords = [
        'hai', 'hain', 'ko', 'ki', 'ka', 'ek', 'aur', 'se', 'me', 'mein', 
        'ap', 'aap', 'mai', 'main', 'ye', 'yeh', 'wo', 'woh', 'kya', 'accha', 
        'theek', 'hum', 'tum', 'aapka', 'mera', 'hamara', 'unka', 'kuch',
        'nahin', 'nahi', 'na', 'ji', 'haan', 'bilkul', 'bas'
    ]
    
    # Common Indian financial terms that might indicate fraud when combined with other terms
    indian_financial_terms = [
        'paisa', 'rupay', 'rupee', 'bank', 'account', 'upi', 'paytm', 'phonepe',
        'gpay', 'google pay', 'aadhar', 'pan', 'kyc', 'otp', 'password', 'pin'
    ]
    
    # Keep alphanumeric and some Hindi transliteration characters
    processed_text = re.sub(r'[^\w\s]', ' ', text)
    processed_text = processed_text.lower()
    processed_text = processed_text.split()
    
    # Remove English stopwords and our custom Hindi stopwords
    all_stopwords = list(stopwords.words('english')) + hindi_stopwords
    processed_text = [ps.lemmatize(word) for word in processed_text 
                     if word not in all_stopwords]
    
    # Add special weight to Indian financial terms by duplicating them
    enhanced_text = []
    for word in processed_text:
        enhanced_text.append(word)
        if word in indian_financial_terms:
            enhanced_text.append(word)  # Add important terms twice for emphasis
    
    processed_text = ' '.join(enhanced_text)
    return processed_text

# Function to validate prediction with rule-based checks
def validate_prediction(text, model_prediction, probability):
    """
    Apply rule-based validation to ensure prediction makes sense
    """
    # Convert text to lowercase for easier pattern matching
    text_lower = text.lower()
    
    # High-risk patterns that strongly indicate fraud
    high_risk_patterns = [
        r'(otp|password|pin|cvv).*(share|tell|send|bata|batao)',
        r'(account|khata).*(block|freeze|band)',
        r'(urgent|emergency|jaldi|turant).*(kyc|verify|update)',
        r'(suspicious|fraud|dhokha).*(transaction|activity|gatividhi)',
        r'(aadhar|pan).*(link|connect|verify)'
    ]
    
    # Low-risk patterns that indicate legitimate calls
    low_risk_patterns = [
        r'(help|support|customer service|madad)',
        r'(thank you|dhanyavaad|shukriya)',
        r'(appointment|schedule|booking|samay)',
        r'(information|details|jankari).*(provide|share)',
        r'(call back|return call|wapas call)'
    ]
    
    # Check for high-risk patterns
    high_risk_found = any(re.search(pattern, text_lower) for pattern in high_risk_patterns)
    
    # Check for low-risk patterns
    low_risk_found = any(re.search(pattern, text_lower) for pattern in low_risk_patterns)
    
    # Override model prediction if necessary
    if high_risk_found and not model_prediction:
        return True, max(0.75, probability)  # Override to fraud with higher confidence
    elif low_risk_found and model_prediction:
        return False, max(0.75, 1-probability)  # Override to legitimate with higher confidence
    
    # If no override, return original prediction
    return model_prediction, probability

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Fraud Call Detection API is running. Use /predict endpoint to analyze call transcripts from India."}

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly")
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResult)
async def predict(call: CallTranscript):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not call.text or call.text.strip() == "":
        raise HTTPException(status_code=400, detail="Empty call transcript")
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(call.text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text]).toarray()
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][1]
        
        # Validate prediction with rule-based checks
        is_fraud, confidence = validate_prediction(call.text, bool(prediction), probability)
        
        # Format result
        if is_fraud:
            message = f"FRAUD ALERT: This call is likely fraudulent (Confidence: {confidence:.2%})"
        else:
            message = f"LEGITIMATE: This call appears to be legitimate (Confidence: {confidence:.2%})"
        
        return PredictionResult(
            is_fraud=is_fraud,
            confidence=float(confidence),
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the API server
if __name__ == "__main__":
    uvicorn.run("app_api:app", host="127.0.0.1", port=8000, reload=True)