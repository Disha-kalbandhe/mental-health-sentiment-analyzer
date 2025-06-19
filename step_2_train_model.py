import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the cleaned dataset
df = pd.read_csv("data/merged_clean_dataset.csv")

# Keep only valid labels
df = df[df['label'].isin(['suicidal', 'non-suicidal'])]

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "data/best_sentiment_model.pkl")
joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")
print("✅ Model and Vectorizer saved in data/")
