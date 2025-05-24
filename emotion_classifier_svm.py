import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the extracted features
df = pd.read_csv("ravdess_features.csv")

# Step 2: Convert features from string to NumPy array
X = np.array(df["features"].apply(eval).tolist())
y = df["label"]

# Step 3: Encode emotion labels into numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Build an SVM model with feature scaling
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
clf.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Step 8: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
import joblib

# Save the trained model and the label encoder
joblib.dump(clf, "svm_emotion_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("📁 Model and LabelEncoder saved successfully.")
