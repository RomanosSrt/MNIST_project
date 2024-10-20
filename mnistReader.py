import csv
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical


def showDigits(destination, number):
    check = str(number)
    with open(destination, "r") as csv_file:
        csvreader = csv.reader(csv_file)
        for data in csvreader:
            label = data[0]
            pixels = data[1:]        
            pixels = np.array(pixels, dtype = 'int64')
            pixels = pixels.reshape((28, 28))
            if label == check:
                return label, pixels
                
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





for i in range(10):
    plt.subplot(2, 5, i+1)
    label, pixels = showDigits(dest, i)


    plt.imshow(pixels, cmap='gray', interpolation='none')
    plt.title("Class {}".format(label))
plt.tight_layout()
plt.show()