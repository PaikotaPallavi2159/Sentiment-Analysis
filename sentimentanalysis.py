# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 2. Sample dataset
data = {
    "review": [
        "I love this product, it's amazing!",
        "Very bad experience, not recommended.",
        "Totally worth the price. Satisfied!",
        "Terrible. I want my money back!",
        "Excellent quality and fast delivery.",
        "Worst purchase ever.",
        "Really happy with the service.",
        "Disappointed and frustrated.",
        "Great value for money!",
        "Not what I expected at all."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

# 3. Convert to DataFrame
df = pd.DataFrame(data)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# 5. TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
