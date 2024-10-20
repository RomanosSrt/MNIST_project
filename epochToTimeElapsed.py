import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
import time

# Loading MNIST dataset from TensorFlow
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing the data
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255.0

# One-hot encode the labels
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# List of epochs for the selected networks (10 and 100 epochs)
epochs_to_record = [22, 100]

def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)), 
        Dropout(0.3),    
        Dense(64, activation='relu'), 
        Dropout(0.3),    
        Dense(32, activation='relu'),  
        Dropout(0.3),    
        Dense(10, activation='softmax')
    ])
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=7500,
        decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Lists to store accuracies and times for each network
all_val_accuracies = []
all_epoch_times = []

# Train networks with 10 and 100 epochs
for num_epochs in epochs_to_record:
    print(f"Training network for {num_epochs} epochs...")
    
    model = create_model()
    
    epoch_durations = []
    val_accuracies = []

    class TimeHistory(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            duration = time.time() - self.epoch_start_time
            epoch_durations.append(duration)
            val_accuracies.append(logs['val_accuracy'])

    time_callback = TimeHistory()

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
              batch_size=128, epochs=num_epochs, verbose=0, callbacks=[time_callback])

    all_val_accuracies.append(val_accuracies)
    all_epoch_times.append(epoch_durations)

# Plot accuracy vs epochs for 10 and 100 epoch networks
plt.figure(figsize=(12, 6))
for i, num_epochs in enumerate(epochs_to_record):
    plt.plot(range(1, num_epochs + 1), all_val_accuracies[i], label=f'{num_epochs} Epochs')
plt.title('Validation Accuracy vs Epochs for 22 and 100 Epoch Networks')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot accuracy vs time for 10 and 100 epoch networks
plt.figure(figsize=(12, 6))
for i, num_epochs in enumerate(epochs_to_record):
    cumulative_time = np.cumsum(all_epoch_times[i])  # Calculate cumulative time
    plt.plot(cumulative_time, all_val_accuracies[i], label=f'{num_epochs} Epochs')
plt.title('Validation Accuracy vs Time for 22 and 100 Epoch Networks')
plt.xlabel('Cumulative Time (seconds)')
plt.ylabel('Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

