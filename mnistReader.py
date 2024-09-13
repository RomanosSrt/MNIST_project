with open('Images\\test.csv', "r") as test, open('Images\\train.csv', "r") as train:
    trainList = train.readlines()
    testList = test.readlines()
    
print(str(len(trainList)) + " " + str(len(testList)))