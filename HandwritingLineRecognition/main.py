from Model import *
import editdistance

from Processor import wer


# dataLoader = DataLoader('data/', 100, (800, 64), 0.95)
from dataLoader import DataLoader


class FilePaths:
    fnTrain = 'data/'


def train(model, dataLoader):
    """ Train the neural network """
    currentEpoch = 0
    bestCharErrorRate = float('inf')  # Best character error rate
    noImprovementSince = 0  # Number of epochs with no improvement of character error rate
    earlyStopping = 25  # Stop training after this number of epochs with no improvement
    currentBatch = 0
    noEpochsMax = 50

    while currentEpoch < noEpochsMax:
        currentEpoch += 1
        print('Epoch:', currentEpoch)

        # Train
        print('Train neural network')
        dataLoader.trainSet()

        while dataLoader.hasNext():
            currentBatch += 1
            iteratorInfo = dataLoader.getIteratorInfo()
            batch = dataLoader.getNext()
            lossValueForCurrentBatch = model.trainBatch(batch, currentBatch)
            print('Batch:', iteratorInfo[0], '/', iteratorInfo[1], 'Loss:', lossValueForCurrentBatch)

        # Validate(test)
        charErrorRate, addressAccuracy, wordErrorRate = validate(model, dataLoader)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break


def validate(model, loader):
    """ Validate neural network """
    print('Validate neural network')
    loader.validationSet()
    noCharactersErr = 0
    noCharacters = 0
    noWordsOK = 0
    noWords = 0

    totalCER = []  # total character error rate
    totalWER = []  # total word error rate
    while loader.hasNext():  # parse test set
        iteratorInfo = loader.getIteratorInfo()
        print('Batch:', iteratorInfo[0], '/', iteratorInfo[1])
        batch = loader.getNext()
        recognizedTexts = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognizedTexts)):
            noWordsOK += 1 if batch.groundTruthTexts[i] == recognizedTexts[i] else 0
            noWords += 1

            # quantify how dissimilar are 2 strings
            # count the number of minimum operations required to transform one string into another
            dist = editdistance.eval(recognizedTexts[i], batch.groundTruthTexts[i])

            currCER = dist / max(len(recognizedTexts[i]), len(batch.groundTruthTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognizedTexts[i].split(), batch.groundTruthTexts[i].split())
            totalWER.append(currWER)

            noCharactersErr += dist
            noCharacters += len(batch.groundTruthTexts[i])

            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.groundTruthTexts[i] + '"', '->', '"' + recognizedTexts[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER) / len(totalCER)
    accuracy = noWordsOK / noWords
    wordErrorRate = sum(totalWER) / len(totalWER)
    print('Character error rate: %f%%. Accuracy: %f%%. Word error rate: %f%%' %
          (charErrorRate * 100.0, accuracy * 100.0, wordErrorRate * 100.0))
    return charErrorRate, accuracy, wordErrorRate


def main():
    loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imageSize, 0.95, Model.maxTextLen)
    model = Model(loader.characters)
    train(model, loader)  # train and validate after each epoch


if __name__ == '__main__':
    main()
