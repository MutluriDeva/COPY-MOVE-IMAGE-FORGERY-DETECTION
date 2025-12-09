import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set dataset paths
DATASET_PATH = "/content/mic/MICC-F220"
AU_PATH = os.path.join(DATASET_PATH, "Au")
TU_PATH = os.path.join(DATASET_PATH, "Tu")
GROUND_TRUTH_FILE = os.path.join(DATASET_PATH, "groundtruthDB_220.txt")

# Load Ground Truth Labels
def load_groundtruth(file_path):
    labels = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, label = parts[0].strip().lower(), int(parts[1].strip())
                labels[img_name] = label
    return labels

groundtruth = load_groundtruth(GROUND_TRUTH_FILE)

# CenSurE + FREAK Feature Extraction
def extract_keypoints_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.xfeatures2d.StarDetector_create()
    keypoints = detector.detect(gray, None)
    freak = cv2.xfeatures2d.FREAK_create()
    keypoints, descriptors = freak.compute(gray, keypoints)
    if descriptors is None:
        return np.zeros((1024, 64))
    return descriptors[:1024] if descriptors.shape[0] > 1024 else np.pad(descriptors, ((0, 1024 - descriptors.shape[0]), (0, 0)), mode='constant')

# CNN Feature Extraction using VGG16
def extract_cnn_features(image):
    image_resized = cv2.resize(image, (224, 224)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=vgg16.input, outputs=Flatten()(vgg16.output))
    features = model.predict(image_resized, verbose=0)
    return features.flatten()

# Process Dataset
def process_dataset(folder_path, groundtruth):
    dataset = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        descriptors = extract_keypoints_descriptors(image)
        cnn_features = extract_cnn_features(image)
        label = groundtruth.get(img_name.strip().lower(), None)
        if label is None:
            continue
        dataset.append((descriptors, cnn_features, label))
    return dataset

# Load Full Data
au_data = process_dataset(AU_PATH, groundtruth)
tu_data = process_dataset(TU_PATH, groundtruth)
all_data = au_data + tu_data

# Split into 80% train + 20% test
train_data, test_data = train_test_split(
    all_data, test_size=0.2, random_state=42,
    stratify=[label for _, _, label in all_data]
)

# Prepare Training Arrays
X_keypoints = np.array([desc for desc, _, _ in train_data]).reshape(len(train_data), -1)
X_cnn = np.array([cnn_feat for _, cnn_feat, _ in train_data])
y_labels = np.array([label for _, _, label in train_data])

# Model Definition
input_kp = Input(shape=(X_keypoints.shape[1],))
input_cnn = Input(shape=(X_cnn.shape[1],))
merged = Concatenate()([BatchNormalization()(input_kp), BatchNormalization()(input_cnn)])
dense1 = Dense(512, activation="relu")(merged)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(256, activation="relu")(drop1)
drop2 = Dropout(0.3)(dense2)
output = Dense(1, activation="sigmoid")(drop2)

fusion_model = Model(inputs=[input_kp, input_cnn], outputs=output)
fusion_model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train
fusion_model.fit([X_keypoints, X_cnn], y_labels, epochs=20, batch_size=16, validation_split=0.2)

# Save Model
fusion_model.save("censure_cnn_forgery_model.h5")

# Prepare Test Arrays (20%)
X_test_keypoints = np.array([desc for desc, _, _ in test_data]).reshape(len(test_data), -1)
X_test_cnn = np.array([cnn_feat for _, cnn_feat, _ in test_data])
y_test = np.array([label for _, _, label in test_data])

# Predict
y_pred_probs = fusion_model.predict([X_test_keypoints, X_test_cnn])
y_pred = (y_pred_probs > 0.5).astype(int)

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = np.mean(y_pred == y_test)

# Print Results
print(f"📌 Dataset: MICC-F220")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall: {recall:.4f}")
print(f"🔹 F1-Score: {f1*100:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Display
plt.figure(figsize=(4, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Original", "Tampered"])
disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
