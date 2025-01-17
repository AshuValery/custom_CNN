import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Check if GPU is available and use it
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available. Training will run on CPU.")

# Load data from file
# big_data = np.loadtxt("complete_simulation_output_100.txt")
#big_data = data
#num_classes = big_data.shape[0]

# Extract features (X) and labels (y) from big_data
X = big_data[:, :-1]  # Features (all columns except the last one)
y = big_data[:, -1]   # Labels (last column)

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

# Reshape input data to fit CNN input shape
X_train = X_train.reshape(-1, 150, 150, 1)  # Assuming your images are 150x150 grayscale
X_val = X_val.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)

# Fit the generator to the training data
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define CNN architecture
model = models.Sequential([
    Input(shape=(150, 150, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Assuming num_classes is the number of unique labels
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
start_time = time.time()
# Train the model
history = model.fit(X_train, y_train, batch_size=64,
                    epochs=20,
                    validation_data=(X_val, y_val))
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# Evaluate the model on the test set
start_time2 = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)
end_time2 = time.time()
print(f"Inference time: {end_time2 - start_time2} seconds")


# Calculate Top-1 accuracy
top_1_correct = np.sum(np.diag(conf_matrix))  # Correct predictions on the diagonal
top_1_total = np.sum(conf_matrix)  # Total number of samples
top_1_accuracy = top_1_correct / top_1_total

# Calculate Top-3 accuracy
top_3_correct = 0
for i in range(len(conf_matrix)):
    # Get the row for the true class 'i' and sort by prediction counts
    sorted_row = np.argsort(conf_matrix[i])[::-1]  # Sort in descending order
    # Check if the true class is in the top 3 predicted classes
    if i in sorted_row[:3]:
        top_3_correct += np.sum(conf_matrix[i, sorted_row[:3]])

top_3_accuracy = top_3_correct / top_1_total

print(f"Total: {top_1_total} ")
print(f"Top-1 Accuracy: {top_1_accuracy * 100:.2f}%")
print(f"Top-3 Accuracy: {top_3_accuracy * 100:.2f}%")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_0t.png')
plt.close()
