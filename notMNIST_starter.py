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


#################################################################



# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']

    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    # train_features = train_features.astype(np.float32) / 255.0
    # train_features = train_features.reshape(train_features.shape[0], -1)
    train_features_flatten = []
    for train_feature in train_features:
        vec = np.array(train_feature).flatten()
        vec = vec / 255.0
        vec = cv(vec)
        train_features_flatten.append(vec)

    train_labels = [onehot(i, 10) for i in train_labels]

    test_features_flatten = []
    for test_feature in test_features:
        vec = np.array(test_feature).flatten()
        vec = vec / 255.0
        vec = cv(vec)
        test_features_flatten.append(vec)

    trainingData = zip(train_features_flatten, train_labels)
    testingData = zip(test_features_flatten, test_labels)
    return (trainingData, testingData)

###################################################################


trainingData, testingData = prepData()

net = network.Network([784, 30, 10])
net.SGD(trainingData, 20, 10, 3, test_data=testingData)
network.saveToFile(net, "part2.pkl")