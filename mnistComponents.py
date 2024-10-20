with open('Images\\test.csv', "r") as test, open('Images\\train.csv', "r") as train:
    trainList = train.readlines()
    testList = test.readlines()


def countClasses(set):
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for image in set:
        if (image[0] == "0"):
            t0 += 1
        elif (image[0] == "1"):
            t1 += 1
        elif (image[0] == "2"):
            t2 += 1
        elif (image[0] == "3"):
            t3 += 1
        elif (image[0] == "4"):
            t4 += 1
        elif (image[0] == "5"):
            t5 += 1
        elif (image[0] == "6"):
            t6 += 1
        elif (image[0] == "7"):
            t7 += 1
        elif (image[0] == "8"):
            t8 += 1
        elif (image[0] == "9"):
            t9 += 1
            
            
    print("0 x " , t0)
    print("1 x " , t1)
    print("2 x " , t2)
    print("3 x " , t3)
    print("4 x " , t4)
    print("5 x " , t5)
    print("6 x " , t6)
    print("7 x " , t7)
    print("8 x " , t8)
    print("9 x " , t9)
    
    
print("Training set has " + str(len(trainList)) + " images of:")
countClasses(trainList)
print("Test set has " + str(len(testList)) + " images of:")
countClasses(testList)