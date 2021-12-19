import random
import numpy as np
import cv2


def processImage(image, imageSize, enhance=False, dataAugmentation=False):
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


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    WER = Word Error Rate
    Levenshtein distance = the minimum number of single-character(substitution, insertion, deletion) edits required to change one word into the other
        e.g. kitten -> sitting
            1. kitten -> sitten (substitution k -> s)
            2. sitten -> sittin (substitution e -> i)
            3. sittin -> sitting (insertion g)

    """
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]