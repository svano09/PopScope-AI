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

print("🚀 TRAIN NETFLIX START (ENSEMBLE 3 MODELS)")

#  PATH 
os.makedirs("models/netflix", exist_ok=True)

#  LOAD 
df = pd.read_csv("data/netflix_titles.csv")

#  CLEAN 
df = df.dropna(subset=["type", "release_year", "duration", "listed_in"])

#  FEATURE ENGINEERING 
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)

# แปลง Genre เป็น 0 กับ 1
genres_df = df["listed_in"].str.get_dummies(sep=', ')
genre_columns = list(genres_df.columns)
df = pd.concat([df, genres_df], axis=1)

np.random.seed(42)
base_score = np.random.normal(50, 15, len(df)) # คะแนนตั้งต้น
type_bonus = np.where(df["type"] == "TV Show", 10, 0)
year_bonus = (df["release_year"] - df["release_year"].min()) * 0.5
df["synthetic_score"] = base_score + type_bonus + year_bonus

# กำหนดว่าถ้าคะแนนรวมเกิน Percentile ที่ 70 คือ Popular (1) นอกนั้น (0)
threshold = df["synthetic_score"].quantile(0.7)
df["popular"] = (df["synthetic_score"] > threshold).astype(int)

#  ENCODE 
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])
joblib.dump(le, "models/netflix/type_encoder.pkl")

#  FEATURES 
feature_cols = ["type_encoded", "release_year", "duration_num"] + genre_columns
joblib.dump(genre_columns, "models/netflix/genre_columns.pkl") 

X = df[feature_cols]
y = df["popular"]

#  SCALE 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/netflix/scaler.pkl")

#  SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#  ML
# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
# 3. Logistic Regression
lr = LogisticRegression(random_state=42)

# สร้าง Ensemble Model 
ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)
joblib.dump(ensemble_model, "models/netflix/model.pkl")

#  NN 
nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.3), # ป้องกัน Overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
nn.save("models/netflix/nn.keras")

#  RESULT 
acc = ensemble_model.score(X_test, y_test)
print(f"✅ NETFLIX ENSEMBLE DONE | Accuracy: {round(acc*100, 2)}%")
print(f"📌 Total Features trained: {X_scaled.shape[1]}")