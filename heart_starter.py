import csv
from random import sample
import numpy as np
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    


##############################################


# given a list with the features and label for a sample (row of the csv),
# converts it to a numeric feature vector and an integer label
# returns the tuple (feature, label)
def getDataFromSample(sample):

    # sbp
    sbp = cv([standardize(int(sample[1]), 138.3, 20.5)])

    # tobacco
    tobacco = cv([standardize(float(sample[2]), 3.64, 4.59)])

    # ldl
    ldl = cv([standardize(float(sample[3]), 4.74, 2.07)])

    # adiposity
    adiposity = cv([standardize(float(sample[4]), 25.4, 7.77)])

    # famhist
    if(sample[5] == 'Present'):
        famhist = cv([1])
    else:
        famhist = cv([0])

    # typea
    typea = cv([standardize(int(sample[6]), 53.1, 9.81)])

    # obesity
    obesity = cv([standardize(float(sample[7]), 26.0, 4.21)])

    # alcohol
    alcohol = cv([standardize(float(sample[8]), 17.0, 24.5)])

    # age
    # age = cv([standardize(int(sample[9]), 42.8, 14.6)])
    age = cv([int(sample[9]) * 1.0 / 64.0])

    features = np.concatenate((sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age), axis=0)

    # chd (label)
    label = int(sample[10])

    return (features, label)

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)
        next(reader, None)  # skip the header row

        n = 0
        features = []
        labels = []

        for row in reader:
            featureVec, label = getDataFromSample(row)
            features.append(featureVec)
            labels.append(label)
            n = n + 1

    print(f"Number of data points read: {n}")
    
    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    n, features, labels = readData('data/heart.csv')
    
    ntrain = int(n * 5/6)
    ntest = n - ntrain

    # split into training and testing data
    trainingFeatures = features[:ntrain]
    trainingLabels = [onehot(label, 2) for label in labels[:ntrain]]    # training labels should be in onehot form

    print(f"Number of training samples: {ntrain}")

    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]
    print(f"Number of testing samples: {ntest}")

    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)
    return (trainingData, testingData)



###################################################


trainingData, testingData = prepData()

net = network.Network([9,10,2])
net.SGD(trainingData, 10, 10, .1, test_data = testingData)
network.saveToFile(net, "part3.pkl")