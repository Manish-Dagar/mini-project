import os
import platform
import numpy as np
import cv2
import imageio
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Path to extracted dataset
PROJECT_DIR = r"C:\Users\Tony Stark\Desktop\Violence Dataset"
print("Path to dataset files:", PROJECT_DIR)

# Check system info
print(f"Running on: {platform.platform()}")

# Constants
IMG_SIZE = 128
COLOR_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 50

# Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Convert video to frames
def video_to_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    
    while vidcap.isOpened():
        frame_id = int(vidcap.get(1))
        success, image = vidcap.read()
        
        if success and frame_id % 7 == 0:
            augmenter = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(scale=(1.0, 1.2), rotate=(-15, 15)),
                iaa.Multiply((0.8, 1.2))
            ])
            image_aug = augmenter(image=image)
            image_resized = cv2.resize(cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            frames.append(image_resized)
        elif not success:
            break
    
    vidcap.release()
    return frames

# Load dataset
def load_dataset():
    classes = ["NonViolence", "Violence"]
    X, y = [], []

    for category in classes:
        class_index = classes.index(category)
        video_dir = os.path.join(PROJECT_DIR, category)

        if not os.path.exists(video_dir):
            print(f"Error: Path not found for {category} at {video_dir}")
            continue

        videos = os.listdir(video_dir)[:350]

        for video in videos:
            video_path = os.path.join(video_dir, video)
            frames = video_to_frames(video_path)
            for frame in frames:
                X.append(frame)
                y.append(class_index)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Error: No data loaded. Please check your dataset.")

    X = np.array(X).reshape(-1, IMG_SIZE * IMG_SIZE * 3)
    y = np.array(y)
    return X, y

# Load data
X_original, y_original = load_dataset()

# Train-test split
stratified_sample = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=73)
for train_idx, test_idx in stratified_sample.split(X_original, y_original):
    X_train, X_test = X_original[train_idx], X_original[test_idx]
    y_train, y_test = y_original[train_idx], y_original[test_idx]

# Reshape and normalize
X_train_nn = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
X_test_nn = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0

# Define model
def build_model():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, COLOR_CHANNELS))
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_tensor=input_tensor)
    
    head_model = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001))(base_model.output)
    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = build_model()
model.summary()

# Callbacks
class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") >= 0.999:
            print("\nAccuracy limit reached, stopping training!")
            self.model.stop_training = True

def lr_schedule(epoch):
    return 0.00001 if epoch < 5 else 0.00005 * 0.8**(epoch - 5)

callbacks = [
    StopTrainingCallback(),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint("ModelWeights.weights.h5", save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=2, mode="min"),
    TensorBoard(log_dir="logs/fit")
]

# Train model
history = model.fit(
    X_train_nn, y_train,
    validation_data=(X_test_nn, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Load best model
print("\nRestoring best weights...")
model.load_weights("ModelWeights.weights.h5")

# Evaluation
def plot_metrics(history):
    metrics_list = ["loss", "accuracy"]
    for metric in metrics_list:
        plt.figure()
        plt.plot(history.history[metric], label="Training")
        plt.plot(history.history[f"val_{metric}"], label="Validation")
        plt.legend()
        plt.title(f"Training & Validation {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.show()

plot_metrics(history)

# Predictions
y_pred = (model.predict(X_test_nn) > 0.5).astype("int32")

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(metrics.classification_report(y_test, y_pred, target_names=["NonViolence", "Violence"]))
