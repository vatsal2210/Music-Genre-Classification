# Source: https://data-flair.training/blogs/python-project-music-genre-classification/

from python_speech_features import mfcc
import scipy.io.wavefile as wav
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random
import operator

import math

# define a function to get the distance between feature vectors and find neighbors:
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(
            instance, trainingSet[x], k
        )
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# identify the nearest neighbors
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.item(), key=operator.index(1), reverse=true)
    return sorter[0][0]


# define a function for model evaluation
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(testSet)


# Extract features from the dataset and dump these features into a binary .dat file “my.dat”:
directory = "__path_to_dataset__"
f = open("my.dat", "wb")
i = 0
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory + folder):
        (rate, sig) = wav.read(directory + folder + "/" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

# Train and test split on the dataset:
dataset = []


def loadDataset(filename, split, trSet, teSet):
    with open("my.dat", "rb") as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset("my.dat", 0.66, trainingSet, testSet)

# KNN prediction
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))
accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)
