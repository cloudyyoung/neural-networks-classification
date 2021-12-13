import numpy as np
import idx2numpy
import network

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])

# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T

# creates a (size,1) array of zeros, whose ith entry is equal to 1
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big')  # number of entries

    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()

    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    
    images = idx2numpy.convert_from_file(imagefile)
    print("Image data:", len(images), "entries")

    
    features = []

    for image in images:
        vec = np.array(image).flatten()
        vec = vec / 255.0
        vec = cv(vec)

        features.append(vec)

    return features


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    _, training_labels = getLabels(trainingLabelFile)
    _, testing_labels = getLabels(testingLabelFile)
    
    training_labels_onehot = []
    for training_label in training_labels:
        training_labels_onehot.append(onehot(training_label, 10))
    
    training_features = getImgData(trainingImageFile)
    testing_features = getImgData(testingImageFile)

    trainingData = zip(training_features, training_labels_onehot)
    testingData = zip(testing_features, testing_labels)
    return (trainingData, testingData)


###################################################


trainingData, testingData = prepData()


net = network.Network([9, 10, 2])
net.SGD(trainingData, 10, 10, .1, test_data=testingData)
network.saveToFile(net, "part1.pkl")
