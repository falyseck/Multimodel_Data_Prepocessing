import pandas as pd
import joblib
import random

# Load models
face_model = joblib.load('models/face_model.pkl')
voice_model = joblib.load('models/voice_model.pkl')
product_model = joblib.load('models/product_model.pkl')

# Load datasets
image_df = pd.read_csv('data/processed/image_features.csv')
audio_df = pd.read_csv('data/processed/audio_features.csv')
merged_df = pd.read_csv('data/processed/merged_customer_data.csv')

print("🔐 SYSTEM STARTED")

# -----------------------------
# STEP 1: Simulate face input
# -----------------------------
face_sample = image_df.sample(1)
face_X = face_sample.drop(columns= ['person','image'], axis=1)
true_face_user = face_sample['person'].values[0]

predicted_face_user = face_model.predict(face_X)[0]

print("\n📸 Face detected:", predicted_face_user)

# -----------------------------
# STEP 2: Simulate voice input
# -----------------------------

# For successful demo
voice_sample = audio_df[audio_df['person'] == predicted_face_user].sample(1)
# For random test (unauthorized)
#voice_sample = audio_df.sample(1)
voice_X = voice_sample.drop('person', axis=1)
true_voice_user = voice_sample['person'].values[0]

predicted_voice_user = voice_model.predict(voice_X)[0]

print("🎤 Voice detected:", predicted_voice_user)

# -----------------------------
# STEP 3: Authorization check
# -----------------------------
if predicted_face_user == predicted_voice_user:
    print("\n✅ User verified:", predicted_face_user)

    # Get user data for recommendation
    user_data = merged_df.sample(1)  # simulate user profile

    X_product = user_data.drop(columns=['product_category','customer_id','purchase_date'], axis=1)

    predicted_product = product_model.predict(X_product)[0]

    print("🛒 Recommended product:", predicted_product)

else:
    print("\n❌ UNAUTHORIZED ACCESS")