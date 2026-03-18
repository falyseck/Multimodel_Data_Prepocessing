import pandas as pd
import joblib

# Load models
face_model = joblib.load('models/face_model.pkl')
voice_model = joblib.load('models/voice_model.pkl')
product_model = joblib.load('models/product_model.pkl')

# Load datasets
image_df = pd.read_csv('data/processed/image_features.csv')
audio_df = pd.read_csv('data/processed/audio_features.csv')
merged_df = pd.read_csv('data/processed/merged_customer_data.csv')

print("🔐 MULTIMODAL AUTHENTICATION SYSTEM 🔐")

# -----------------------------
# STEP 1: User inputs face
# -----------------------------
face_input = input("\n📸 Enter name for FACE (faly / ivan / mahad / duba): ").lower()

# Find matching face sample
face_sample = image_df[image_df['person'] == face_input]

if face_sample.empty:
    print("❌ Face not recognized!")
    exit()

face_sample = face_sample.sample(1)
face_X = face_sample.drop(columns=['person','image'], axis=1)

predicted_face_user = face_model.predict(face_X)[0]
print("📸 Face detected as:", predicted_face_user)

# -----------------------------
# STEP 2: User inputs voice
# -----------------------------
voice_input = input("\n🎤 Enter name for VOICE (faly / ivan / mahad / duba): ").lower()

# Find matching voice sample
voice_sample = audio_df[audio_df['person'] == voice_input]

if voice_sample.empty:
    print("❌ Voice not recognized!")
    exit()

voice_sample = voice_sample.sample(1)
voice_X = voice_sample.drop('person', axis=1)

predicted_voice_user = voice_model.predict(voice_X)[0]
print("🎤 Voice detected as:", predicted_voice_user)

# -----------------------------
# STEP 3: Authorization
# -----------------------------
if predicted_face_user == predicted_voice_user:
    print("\n✅ ACCESS GRANTED")

    # Simulate user data
    user_data = merged_df.sample(1)

    X_product = user_data.drop(columns=['product_category','purchase_date', 'customer_id'], axis=1)

    predicted_product = product_model.predict(X_product)[0]

    print("🛒 Recommended product:", predicted_product)

else:
    print("\n❌ ACCESS DENIED - UNAUTHORIZED USER")