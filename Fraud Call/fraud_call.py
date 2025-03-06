import re
import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Verify dataset path
dataset_path = "D:\\Python\\AIML\\Fraud Call\\data\\fraud_call.file"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Load the dataset
data = pd.read_csv(dataset_path, sep='\t', names=['label','content'], on_bad_lines='skip')

data.head()

data['label'].value_counts()

import seaborn as sns

sns.countplot(x='label', data=data)

import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
ps = WordNetLemmatizer()
cv = TfidfVectorizer(max_features=500)

def remove_digit(data) :
    corpos = []
    for i in range(0, len(data)) :
        review = re.sub('[^a-zA-Z]', ' ', data['content'][i])
        review = review.lower()
        review = review.split()
        review = [ps.lemmatize(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpos.append(review)
    return corpos

from sklearn.metrics import recall_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

def detect_model(corpos, data):
    x = cv.fit_transform(corpos).toarray()
    y = pd.get_dummies(data['label'])
    y = y.iloc[:, 1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    fraud_detect = MultinomialNB().fit(x_train, y_train)
    print("model has trained.")

    y_ped = fraud_detect.predict(x_test)
    cong_m = confusion_matrix(y_test, y_ped)
    acc = accuracy_score(y_test, y_ped)
    recall = recall_score(y_test,y_ped)
    cl_r = classification_report(y_test,y_ped)

    print("Confusion matrix:\n", cong_m)
    print("Accuracy_score:", acc)
    print("recall_score is:",recall)
    print("Classification report:\n",cl_r)
    
    return fraud_detect

list = remove_digit(data)
model= detect_model(list, data)

# Save the trained model and vectorizer
import pickle

# Create models directory if it doesn't exist
models_dir = "D:\\Python\\AIML\\Fraud Call\\models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save model
model_path = os.path.join(models_dir, "fraud_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# Save vectorizer
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
with open(vectorizer_path, 'wb') as f:
    pickle.dump(cv, f)
print(f"Vectorizer saved to {vectorizer_path}")

# Demo prediction
print("\nDemo prediction:")
sample_text = "Hello, this is your bank calling about your account security."

# Preprocess the sample text
processed_text = re.sub('[^a-zA-Z]', ' ', sample_text)
processed_text = processed_text.lower()
processed_text = processed_text.split()
processed_text = [ps.lemmatize(word) for word in processed_text if word not in stopwords.words('english')]
processed_text = ' '.join(processed_text)

# Vectorize and predict
text_vector = cv.transform([processed_text]).toarray()
prediction = model.predict(text_vector)[0]
probability = model.predict_proba(text_vector)[0][1]

if prediction == 1:
    print(f"⚠️ FRAUD ALERT: This call is likely fraudulent (Confidence: {probability:.2%})")
else:
    print(f"✅ LEGITIMATE: This call appears to be legitimate (Confidence: {(1-probability):.2%})")

print("\nModel training and evaluation complete!")

