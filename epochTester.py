import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt

# Function to load data
def loadData(destination):
    with open(destination, "r") as csv_file:
        reader = csv.reader(csv_file)
        data = np.array(list(reader)).astype(float)
    X = data[:, 1:]  # Pixel values
    Y = data[:, 0]   # Labels
    X = X / 255.0    # From grayscale values of 0-255 to absolute values 0-1
    Y = to_categorical(Y, num_classes=10)
    return X, Y

# Loading data
dest = 'Images\\train.csv'
dest2 = 'Images\\test.csv'
X_train, Y_train = loadData(dest)
X_test, Y_test = loadData(dest2)

# List to store results for each epoch
epochs_to_record = range(10, 101, 10)  # Every 10th epoch from 10 to 100
val_losses = []
val_accuracies = []

# Model architecture
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)), 
        Dropout(0.3),    
        Dense(64, activation='relu'), 
        Dropout(0.3),    
        Dense(32, activation='relu'),  
        Dropout(0.3),    
        Dense(10, activation='softmax')  # Output layer with softmax activation
    ])
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=7500,
        decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Training the model for each epoch from 10 to 100
for epochs in epochs_to_record:
    print(f"Training model for {epochs} epochs...")
    
    # Create a new model for each iteration to ensure it starts fresh
    model = create_model()
    
    # Train model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                        batch_size=128, epochs=epochs, verbose=0)
    
    # Get the validation loss and accuracy for the last epoch in this range
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

# Plotting the validation loss and accuracy
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Validation Loss over epochs
axs[0].plot(epochs_to_record, val_losses, marker='o', color='blue', label='Validation Loss')
axs[0].set_title('Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Validation Loss')
axs[0].legend(loc='upper right')

# Subplot 2: Validation Accuracy over epochs
axs[1].plot(epochs_to_record, val_accuracies, marker='o', color='green', label='Validation Accuracy')
axs[1].set_title('Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Validation Accuracy')
axs[1].legend(loc='upper left')

# Adjust layout
plt.tight_layout()
plt.show()
