import os.path
import random
import numpy as np
import cv2


class DatasetSample:  # corresponds to one image; representation: (groundTruthText = actual text, filePath)
    def __init__(self, groundTruthText, filePath):
        self.groundTruthText = groundTruthText
        self.filePath = filePath


class Batch:
    def __init__(self, groundTruthTexts, images):
        self.images = images
        self.groundTruthTexts = groundTruthTexts


class DataLoader:
    def __init__(self, filesPath, batchSize, imageSize, dataSplitPercentage):
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.samples = []
        self.currentIndex = 0

        # file from IAM containing on each line(as described in the beginning of it as comment lines):
        # (line id, result of word segmentation, graylevel, number of components, bounding box(x, y, w, h),
        # transcription = ground truth text)
        # RESULT OF WORD SEGMENTATION: if err -> the segmentation of the line has one or more errors
        # eg: a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
        linesFile = open(filesPath + "lines.txt")

        bad_samples = []

        for line in linesFile:
            if not line or line[0] == '#':  # ignore comment lines
                continue

            lineSplit = line.strip().split(' ')  # remove trailing spaces and split by space the lines.txt file

            # configuration of the images in folders
            # eg: a01/a01-000u/a01-000u-00.png -> part1/part1-part2/part1-part2-part3.png -> part1-part2-part3 = line id
            fileNameSplit = lineSplit[0].split('-')  # obtain from the line id: [part1, part2, part3]

            # fileName = data/lines/part1/part1-part2/part1-part2-part3.png
            fileName = filesPath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                       lineSplit[0] + '.png'

            groundTruthTextList = lineSplit[8].split('|')  # the actual text for each line as a list of words
            # print(groundTruthTextList)

            # join the list of words with space to form a line and truncate the line if
            groundTruthText = ' '.join(groundTruthTextList)

            # check if image is not empty
            if not os.path.getsize(fileName):
                print("bad sample:" + fileName)
                bad_samples.append(lineSplit[0] + '.png')
                continue

            self.samples.append(DatasetSample(groundTruthText, fileName))

        # split the images in training and testing sets by a given percentage
        # eg. if dataSplitPercentage = 0.95, then 95% of the images will be for training and 5% for testing
        splitIndex = int(dataSplitPercentage * len(self.samples))
        self.trainSamples = self.samples[:splitIndex]
        self.testSamples = self.samples[splitIndex:]

        self.groundTruthTrainLines = [x.groundTruthText for x in self.trainSamples]
        self.groundTruthTestLines = [x.groundTruthText for x in self.testSamples]

        # chosen samples per epoch
        self.numberOfSamplesTrainEpoch = 9500

    def processImage(self, image, imageSize, enhance=False, dataAugmentation=False):
        #  there are damaged files in IAM - use black image instead
        if image is None:
            image = np.zeros([imageSize[1], imageSize[0]])
            print("Image None!")

        # increase dataset size by applying random stretches to the image
        if dataAugmentation:
            stretch = (random.random() - 0.5)
            widthStretched = max(int(image.shape[1] * (1 + stretch)), 1)
            image = cv2.resize(image, (widthStretched, image.shape[0]))

        # increase contrast and line width
        if enhance:
            pxmin = np.min(image)
            pxmax = np.max(image)
            imageContrast = (image - pxmin) / pxmax - pxmin * 255
            kernel = np.ones((3, 3), np.uint8)
            image = cv2.erode(imageContrast, kernel, iterations=1)

        (width, height) = imageSize
        (h, w) = image.shape
        fx = w / width
        fy = h / height
        f = max(fx, fy)
        newSize = (max(min(width, int(w / f)), 1),
                   max(min(height, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
        image = cv2.resize(image, newSize,
                           interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC interpolation best approximate the pixels image
        target = np.ones([height, width]) * 255
        target[0:newSize[1], 0:newSize[0]] = image

        # transpose
        image = cv2.transpose(target)
        return image

    def trainSetIteratorInit(self):
        self.currentIndex = 0
        random.shuffle(self.trainSamples)  # shuffle the samples in each epoch
        self.samples = self.trainSamples

    def testSetIteratorInit(self):
        self.currentIndex = 0
        self.samples = self.testSamples

    def getIteratorInfo(self):
        return self.currentIndex // self.batchSize + 1, len(self.samples) // self.batchSize

    def hasNextIterator(self):
        return self.currentIndex + self.batchSize <= len(self.samples)

    def getNextIterator(self):
        batchRange = range(self.currentIndex, self.currentIndex + self.batchSize)
        gtTexts = [self.samples[i].groundTruthText for i in batchRange]
        images = [self.processImage(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imageSize)
                  for i in batchRange]
        self.currentIndex += self.batchSize
        return Batch(gtTexts, images)

    def getAllBatches(self, samples):
        batches = []
        currentIndex = 0
        while currentIndex <= len(samples):
            batchRange = range(currentIndex, min(len(samples), currentIndex + self.batchSize))
            # print(batchRange)
            groundTruthTexts = [self.samples[i].groundTruthText for i in batchRange]
            images = [self.processImage(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imageSize) for
                      i in batchRange]
            batches.append(Batch(groundTruthTexts, images))
            currentIndex += self.batchSize
        return batches
