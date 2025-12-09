import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set Seeds for Reproducibility
SEED = 45
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATASET_PATH = "/content/mic/MICC-F220"
AU_PATH = os.path.join(DATASET_PATH, "Au")
TU_PATH = os.path.join(DATASET_PATH, "Tu")
GROUND_TRUTH_FILE = os.path.join(DATASET_PATH, "groundtruthDB_220.txt")

# Load Ground Truth
def load_groundtruth(file_path):
    labels = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, label = parts[0].strip().lower(), int(parts[1])
                labels[img_name] = label
    return labels

groundtruth = load_groundtruth(GROUND_TRUTH_FILE)

# VGG16 Feature Extractor
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model = Model(inputs=vgg_base.input, outputs=GlobalAveragePooling2D()(vgg_base.output))

def extract_vgg_features(image):
    resized = cv2.resize(image, (224, 224))
    preprocessed = preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))
    features = vgg_model.predict(preprocessed, verbose=0)
    return features.flatten()

def extract_sift_features(image, max_features=256):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((max_features, 128))
    elif descriptors.shape[0] > max_features:
        descriptors = descriptors[:max_features]
    else:
        descriptors = np.vstack([descriptors, np.zeros((max_features - descriptors.shape[0], 128))])
    return descriptors.flatten()

# Data Augmentation Generator
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_image(image, augmentations=2):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    augmented = []
    for _ in range(augmentations):
        for batch in datagen.flow(image, batch_size=1):
            augmented.append(batch[0].astype(np.uint8))
            break
    return augmented

def process_dataset(folder, label_dict, augment=False, aug_factor=2):
    data = []
    for fname in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            label = label_dict.get(fname.strip().lower(), None)
            if label is None:
                continue

            # Original image features
            vgg_feat = extract_vgg_features(img)
            sift_feat = extract_sift_features(img)
            data.append((vgg_feat, sift_feat, label))

            # Augmented image features
            if augment:
                augmented_imgs = augment_image(img, augmentations=aug_factor)
                for aug_img in augmented_imgs:
                    vgg_feat_aug = extract_vgg_features(aug_img)
                    sift_feat_aug = extract_sift_features(aug_img)
                    data.append((vgg_feat_aug, sift_feat_aug, label))
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return data

# Load and Augment Data
au_data = process_dataset(AU_PATH, groundtruth, augment=True)
tu_data = process_dataset(TU_PATH, groundtruth, augment=True)

# Combine and Shuffle
dataset = au_data + tu_data
random.shuffle(dataset)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=SEED, stratify=[d[2] for d in dataset])

X_vgg = np.array([v for v, _, _ in train_data])
X_sift = np.array([s for _, s, _ in train_data])
y_train = np.array([y for _, _, y in train_data])

X_test_vgg = np.array([v for v, _, _ in test_data])
X_test_sift = np.array([s for _, s, _ in test_data])
y_test = np.array([y for _, _, y in test_data])

# Normalize and Reduce Dimensions
scaler = StandardScaler()
X_sift_scaled = scaler.fit_transform(X_sift)
X_test_sift_scaled = scaler.transform(X_test_sift)

pca = PCA(n_components=128)
X_sift_pca = pca.fit_transform(X_sift_scaled)
X_test_sift_pca = pca.transform(X_test_sift_scaled)

# Build Model
input_vgg = Input(shape=(X_vgg.shape[1],))
input_sift = Input(shape=(X_sift_pca.shape[1],))
merged = Concatenate()([input_vgg, input_sift])
x = Dense(512, activation='relu')(merged)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_vgg, input_sift], outputs=output)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    [X_vgg, X_sift_pca], y_train,
    epochs=30, batch_size=16, validation_split=0.2, shuffle=True
)

# Evaluate
y_probs = model.predict([X_test_vgg, X_test_sift_pca])
y_pred = (y_probs > 0.5).astype(int)

# Metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Accuracy: {accuracy*100:.2f}")
print(f"🎯 F1-Score on Test Set: {f1*100:.2f}")
print(f"✅ Precision: {precision*100:.2f}")
print(f"📌 Recall: {recall*100:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(4, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Original", "Tampered"])
disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Training Curves
plt.figure(figsize=(4, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(history.history['loss'], label='Train Loss', color='orange')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(history.history['val_loss'], label='Validation Loss', color='purple')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
