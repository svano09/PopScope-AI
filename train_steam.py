import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("🚀 TRAIN STEAM START (Final Feature Engineering Mode)")

# 1. PATH 
os.makedirs("models/steam", exist_ok=True)

# 2. LOAD 
df = pd.read_csv("data/steam_games_2026.csv")

# 3. DATA CLEANING 
df = df.dropna(subset=["Price_USD", "Discount_Pct", "Primary_Genre", "Review_Score_Pct"])

# 4. FEATURE ENGINEERING
df["Final_Price"] = df["Price_USD"] - (df["Price_USD"] * (df["Discount_Pct"] / 100))

# 5. TARGET 
df["popular"] = (df["Review_Score_Pct"] > 80).astype(int)

# 6. ENCODE
le = LabelEncoder()
df["Primary_Genre_encoded"] = le.fit_transform(df["Primary_Genre"])
joblib.dump(le, "models/steam/label_encoder.pkl")

# 7. FEATURES 
X = df[["Final_Price", "Discount_Pct", "Primary_Genre_encoded"]]
y = df["popular"]

# 8. SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/steam/scaler.pkl")

# 9. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 10. ML MODELS 
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="soft"
)
ensemble.fit(X_train, y_train)
joblib.dump(ensemble, "models/steam/ensemble.pkl")

# 11. NEURAL NETWORK 
nn = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
nn.save("models/steam/nn.keras")

# RESULT
acc = ensemble.score(X_test, y_test)
print(f"✅ STEAM DONE | Realistic Accuracy: {round(acc*100, 2)}%")