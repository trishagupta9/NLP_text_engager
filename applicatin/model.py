import kagglehub
from networkx import display
from networkx import display

# Download latest version
path = kagglehub.dataset_download("kushagra3204/sentiment-and-emotion-analysis-dataset")

print("Path to dataset files:", path)
import os

# List contents of the new dataset directory
new_dataset_contents = os.listdir(path)
print(f"Contents of the dataset directory: {new_dataset_contents}")
archive_path = os.path.join(path, 'archive')
archive_contents = os.listdir(archive_path)
print(f"Contents of the 'archive' directory: {archive_contents}")
import pandas as pd

data_file_path = os.path.join(archive_path, 'combined_sentiment_data.csv')
df = pd.read_csv(data_file_path)
display(df.head())

import pandas as pd

data_file_path = os.path.join(archive_path, 'combined_sentiment_data.csv')
df = pd.read_csv(data_file_path)
display(df.head())

print("Missing values per column:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

import re

def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

df['cleaned_sentence'] = df['sentence'].apply(clean_text)
print("DataFrame after text cleaning:")
display(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting to 5000 features for demonstration

# Fit and transform the cleaned sentences
X = tfidf_vectorizer.fit_transform(df['cleaned_sentence'])

print("Shape of TF-IDF features (X):")
print(X.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode the target variable 'sentiment' into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

print("Encoded sentiment labels (y):")
print(y[:5]) # Display first 5 encoded labels
print("Original sentiment labels:")
print(df['sentiment'].head().values) # Display first 5 original labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

import joblib
# Save the trained model and the TF-IDF vectorizer
joblib.dump(model, 'trained_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

model.fit(X_train, y_train)


