from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import tldextract

# Load trained model and vectorizer
with open("phishing_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Function to extract URLs from email
def extract_urls(email_text):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
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
    
    url_features_tfidf = vectorizer.transform(urls)
    predictions = model.predict(url_features_tfidf)
    
    return "Phishing Email Detected!" if any(predictions) else "Safe Email."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]
    result = classify_email(email_text)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
