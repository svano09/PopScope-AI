import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("🚀 TRAIN STEAM START")

# ---------------- PATH ----------------
os.makedirs("models/steam", exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv("data/steam_games_2026.csv")

# ---------------- CLEAN ----------------
df = df.dropna(subset=["Price_USD", "Review_Score_Pct", "Primary_Genre"])

# ---------------- TARGET ----------------
df["popular"] = df["Review_Score_Pct"] > 80

# ---------------- ENCODE ----------------
le = LabelEncoder()
df["Primary_Genre"] = le.fit_transform(df["Primary_Genre"])
joblib.dump(le, "models/steam/label_encoder.pkl")

# ---------------- FEATURES ----------------
X = df[["Price_USD", "Review_Score_Pct", "Primary_Genre"]]
y = df["popular"]

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/steam/scaler.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- ML ----------------
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="soft"
)

ensemble.fit(X_train, y_train)
joblib.dump(ensemble, "models/steam/ensemble.pkl")

# ---------------- NN ----------------
nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=10, batch_size=16)

nn.save("models/steam/nn.keras")

# ---------------- RESULT ----------------
acc = ensemble.score(X_test, y_test)
print(f"✅ STEAM DONE | Accuracy: {round(acc,3)}")