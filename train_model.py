import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import clean_text, logger

def train():
    logger.info("Starting model training process...")

    # Load dataset
    try:
        data = pd.read_csv("spam.csv", sep='\t', names=["label", "message"], encoding='latin-1')
        logger.info(f"Dataset loaded successfully. Total records: {len(data)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Preprocessing
    logger.info("Cleaning and preprocessing data...")
    data["message"] = data["message"].apply(clean_text)
    X = data["message"]
    y = data["label"].map({"ham": 0, "spam": 1})

    # Vectorization
    logger.info("Vectorizing text data...")
    vectorizer = CountVectorizer(stop_words="english")
    X_vectorized = vectorizer.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Model Training
    logger.info("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model Training Complete. Accuracy: {accuracy*100:.2f}%")
    
    # Detailed Metrics for Recruiters
    logger.info("\n--- Classification Report ---\n" + classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    logger.info("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Spam'], yticklabels=['Safe', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Spam Detection Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")

    # Save artifacts
    logger.info("Saving model and vectorizer to disk...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    logger.info("All artifacts saved. System is ready for app.py")

if __name__ == "__main__":
    train()
