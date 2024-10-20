import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from matplotlib import pyplot as plt

                
def loadData(destination):
    with open(destination, "r") as csv_file:
        reader = csv.reader(csv_file)
        data = np.array(list(reader)).astype(float)

    X = data[:, 1:]     #Pixel values
    Y = data[:, 0]      #Labels
    X = X / 255.0       #From grayscale values of 0-255 to absolute values 0-1
    Y = to_categorical(Y, num_classes=10)
    return X, Y


        
dest = 'Images\\train.csv'
dest2 = 'Images\\test.csv'


X_train, Y_train = loadData(dest)
X_test, Y_test = loadData(dest2)

drop =0.35

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), Dropout(drop),      #First layer of 512 neurons accepts data from 784 (28x28 pixels) input features with ReLU activation function
    Dense(64, activation='relu'), Dropout(drop),                          #Second layer of 256 neurons with ReLU activation function
    Dense(32, activation='relu'),  Dropout(drop),                          #Third layer of 128 neurons with ReLU activation function 
    Dense(10, activation='softmax')                         #Output layer of 10 neurons with softmax activation function (raw output (logits) into probability distribution for the 10 classes 0-9)
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=7500,
    decay_rate=0.9)                                                     #Varying learning rate

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)     #lr_schedule)        #Adam Optimizer
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)                #Stochastic Gradient Descent
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Loss function cross-entropy for the measurement of performance
#Optimization Adam algorithm adjustment of learning rate 

#model.add(Dropout(0.05))         #Randomly turning off neurons to avoid reliancies on specific neurons

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size= 128, epochs= 100)#, validation_split= 0.01)            #validation split to control overfitting

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)


#predictions = model.predict(X_test)



fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Training & Validation Loss
axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].set_title('Model Loss')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend(loc='upper right')

# Subplot 2: Training & Validation Accuracy
axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].set_title('Model Accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend(loc='upper left')

# Adjust layout
plt.tight_layout()
plt.show()