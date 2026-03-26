import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("🚀 TRAIN NETFLIX START")

# ---------------- PATH ----------------
os.makedirs("models/netflix", exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv("data/netflix_titles.csv")

# ---------------- CLEAN ----------------
df = df.dropna(subset=["type", "release_year", "duration"])

# ---------------- FEATURE ENGINEERING ----------------
df["duration_num"] = df["duration"].str.extract("(\d+)").astype(float)

# ---------------- TARGET ----------------
threshold = df["release_year"].quantile(0.7)
df["popular"] = df["release_year"] > threshold

# ---------------- ENCODE ----------------
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])
joblib.dump(le, "models/netflix/type_encoder.pkl")

# ---------------- FEATURES ----------------
X = df[["type_encoded", "release_year", "duration_num"]]
y = df["popular"]

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/netflix/scaler.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- ML ----------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
joblib.dump(model, "models/netflix/model.pkl")

# ---------------- NN ----------------
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=10, batch_size=16)

nn.save("models/netflix/nn.keras")

# ---------------- RESULT ----------------
acc = model.score(X_test, y_test)
print(f"✅ NETFLIX DONE | Accuracy: {round(acc,3)}")
