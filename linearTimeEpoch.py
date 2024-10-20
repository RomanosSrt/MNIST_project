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

epochs_to_record = range(10, 101, 10)
execution_times = []

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

for epochs in epochs_to_record:
    print(f"Training model for {epochs} epochs...")

    model = create_model()

    start_time = time.time()

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              batch_size=128, epochs=epochs, verbose=0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    execution_times.append(elapsed_time)

plt.figure(figsize=(8, 6))
plt.plot(epochs_to_record, execution_times, marker='o', color='red')
plt.title('Execution Time vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()