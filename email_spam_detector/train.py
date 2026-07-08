import re
import string
import joblib
import pandas as pd
from pathlib import Path

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ===========================================================
# Load Dataset
# ===========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Dataset" / "spam.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# ===========================================================
# Clean Dataset
# ===========================================================

print("\nDataset Shape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

df["Category"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

print("\nClass Distribution")
print(df["Category"].value_counts())

# ===========================================================
# Text Preprocessing
# ===========================================================

stemmer = PorterStemmer()

stop_words = ENGLISH_STOP_WORDS


def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["clean_message"] = df["Message"].apply(preprocess)

# ===========================================================
# Train Test Split
# ===========================================================

X = df["clean_message"]

y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ===========================================================
# TF-IDF
# ===========================================================

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

# ===========================================================
# Models
# ===========================================================

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "Linear SVM":
    LinearSVC(),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
}

best_model = None
best_name = ""
best_score = 0

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    acc = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    print(f"\n{name}")

    print("-" * 40)

    print(f"Accuracy : {acc:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, prediction))

    if f1 > best_score:

        best_score = f1

        best_model = model

        best_name = name

# ===========================================================
# Best Model
# ===========================================================

print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)

print("Model :", best_name)

print("F1 Score :", round(best_score, 4))

prediction = best_model.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

# ===========================================================
# Save Best Model
# ===========================================================

SAVE_DIR = BASE_DIR / "saved_models"

SAVE_DIR.mkdir(exist_ok=True)

joblib.dump(best_model, SAVE_DIR / "spam_model.pkl")

joblib.dump(vectorizer, SAVE_DIR / "tfidf_vectorizer.pkl")

print("\nBest model saved successfully!")

print(SAVE_DIR)