import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    # Remove non-alphabetic characters
    processed_text = re.sub('[^a-zA-Z]', ' ', text)
    processed_text = processed_text.lower()
    processed_text = processed_text.split()
    # Remove stopwords and lemmatize
    processed_text = [lemmatizer.lemmatize(word) for word in processed_text 
                     if word not in stopwords.words('english')]
    processed_text = ' '.join(processed_text)
    return processed_text

# Process dataset for model training
def preprocess_dataset(data):
    corpus = []
    for i in range(len(data)):
        corpus.append(preprocess_text(data['content'][i]))
    return corpus