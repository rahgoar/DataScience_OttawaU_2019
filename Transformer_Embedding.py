#Using a pipeline
from transformers import pipeline

# Load a pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze the sentiment of a text
text = "The service was excellent, and I thoroughly enjoyed my experience."
result = sentiment_analyzer(text)

print(result)
# Example output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Using Encoding
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
texts = ["This product is amazing!", "I hate this service.", "It's okay, nothing special."]
labels = [1, 0, 0] # 1 for positive, 0 for negative/neutral

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(texts)

# Split data for training a classifier
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a simple classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the classifier
predictions = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
