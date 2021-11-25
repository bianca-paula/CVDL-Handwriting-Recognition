from dataLoader import *
import matplotlib.pyplot as plt

dataLoader = DataLoader('data/', 100, (800, 64), 0.95)

trainBatches = dataLoader.getAllBatches(dataLoader.trainSamples)
testBatches = dataLoader.getAllBatches(dataLoader.testSamples)

'''for batch in testBatches:
    print(len(batch.images))
for batch in trainBatches:
    print(len(batch.images))'''
