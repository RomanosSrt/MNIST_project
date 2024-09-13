import os

def convert(imgf, labelf, outf, n):
    with open(imgf, "rb") as f, open(outf, "w") as o, open(labelf, "rb") as l:

        f.read(16)          #Skip first 16 characters for images (additional info)   
        l.read(8)           #Skip first 8 characters for labels (additional info)
        images = []

            
        for i in range(n):
            image = [ord(l.read(1))]            #insert label in first element
            for j in range(28*28):
                image.append(ord(f.read(1)))         #insert pixel value row by row (28 x 28 pixel images)
            images.append(image)

        for image in images:
            o.write(",".join(str(pix) for pix in image) + "\n")
    


    
destTest = 'Images\\test.csv'
if not os.path.exists(destTest):
    convert('Dataset\\t10k-images-idx3-ubyte', 'Dataset\\t10k-labels-idx1-ubyte', destTest, 10000)

destTrain = 'Images\\train.csv'
if ( not os.path.exists(destTrain)):
    convert('Dataset\\train-images-idx3-ubyte', 'Dataset\\train-labels-idx1-ubyte', destTrain, 60000)