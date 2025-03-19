# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import re
import tldextract
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
file_path = "phishing_site_urls.csv"
df = pd.read_csv(file_path)
df['Label'] = df['Label'].map({'bad': 1, 'good': 0})

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['URL'])
y = df['Label']

y= y.fillna(0.0)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Suspicious word presence
suspicious_words = ["verify", "login", "secure", "update", "bank", "account", "password"]
for word in suspicious_words:
    df[f"contains_{word}"] = df["URL"].apply(lambda x: 1 if word in x.lower() else 0)

# Special character counts
special_chars = ["@", "-", "_", ".", "/", "=", "?"]
for char in special_chars:
    df[f"count_{char}"] = df["URL"].apply(lambda x: x.count(char))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Try different numbers of trees in Random Forest
trees = [10, 50, 100, 200, 300]
accuracies = []

for n in trees:
    model = RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Trees: {n}, Accuracy: {acc:.4f}")

with open("phishing_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")

# Plot Accuracy vs. Number of Trees
plt.figure(figsize=(8, 5))
plt.plot(trees, accuracies, marker='o', linestyle='-')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs. Number of Trees")
plt.grid()
plt.show()

def extract_urls(email_text):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # Regex for full URLs
    extracted_urls = re.findall(url_pattern, email_text)

    # Extract potential domain-like words
    words = email_text.split()
    for word in words:
        ext = tldextract.extract(word)
        if ext.domain and ext.suffix:
            extracted_urls.append(word)

    return list(set(extracted_urls))  # Remove duplicates

# Function to classify an email based on its URLs
def classify_email(email_text):
    urls = extract_urls(email_text)
    if not urls:
        return "No URLs found in the email. Cannot classify."

    # Convert extracted URLs to TF-IDF features
    url_features_tfidf = vectorizer.transform(urls)

    # Predict phishing probability
    predictions = model.predict(url_features_tfidf)

    if any(predictions):
        return "Phishing Email Detected!"
    else:
        return "Safe Email."

# Example usage
email_text = "Hello, visit example-phishing.com or check www.suspicious.net for details."
print(classify_email(email_text))

