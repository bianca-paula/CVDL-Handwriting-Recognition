import tensorflow.compat.v1 as tf
import sys


class Model:
    batchSize = 20
    imageSize = (800, 64)
    maxTextLen = 100

    def __init__(self, charactersList):
        tf.disable_v2_behavior()  # some functionalities used from tensorflow are from the previous version of it

        self.snapshotID = 0
        self.charactersList = charactersList

        # create a place in memory where data will be stored later on in a session
        # input image batch
        self.inputImages = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imageSize[0], Model.imageSize[1]))

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        # learning rate will be set in the session
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # Initialize TensorFlow
        (self.session, self.saver) = self.setupTF()

    def setupCNN(self):
        # CNN layers -> extract the important features from the image
        cnnIn4D = tf.expand_dims(input=self.inputImages, axis=3)

        # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        with tf.compat.v1.name_scope('Conv_Pool_1'):
            kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.1))
            conv = tf.nn.conv2d(cnnIn4D, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Second Layer: Conv (5x5) + Pool (1x2) - Output size: 400 x 16 x 128
        with tf.compat.v1.name_scope('Conv_Pool_2'):
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 8 x 128
        with tf.compat.v1.name_scope('Conv_Pool_BN_3'):
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Fourth Layer: Conv (3x3) - Output size: 200 x 8 x 256
        with tf.compat.v1.name_scope('Conv_4'):
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)

        # Fifth Layer: Conv (3x3) + Pool(2x2) - Output size: 100 x 4 x 256
        with tf.compat.v1.name_scope('Conv_Pool_5'):
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
            conv = tf.nn.conv2d(learelu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Sixth Layer: Conv (3x3) + Pool(1x2) + Simple Batch Norm - Output size: 100 x 2 x 512
        with tf.compat.v1.name_scope('Conv_Pool_BN_6'):
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Seventh Layer: Conv (3x3) + Pool (1x2) - Output size: 100 x 1 x 512
        with tf.compat.v1.name_scope('Conv_Pool_7'):
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        self.cnnOut4D = pool

    def setupRNN(self):
        # RNN layers -> the input is the output of the CNN
        # The output is a matrix representing character-scores for each sequence-element

        # Collapse layer to remove dimension 100 x 1 x 512 --> 100 x 512 on axis=2
        rnnIn3D = tf.squeeze(self.cnnOut4D, axis=[2])

        # LSTM = Long Short-Term Memory implementation of RNN
        # 2 stacked layers of LSTM cell used to build RNN
        numHidden = 512
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=numHidden, state_is_tuple=True, name='basic_lstm_cell') for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # Bi-directional RNN
        # BxTxF -> BxTx2H
        ((forward, backward), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3D, dtype=rnnIn3D.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        # Project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        # The IAM dataset consists of 79 different characters, one more character is needed for blank
        # --> Output size: 100x80, 100 time-steps and 80 characters, i.e. 80 characters for each 100 time-steps
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charactersList) + 1], stddev=0.1))
        self.rnnOut3D = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        # CTC loss and decoder
        # CTC operation tries all possible alignments of the GT text in the image and takes the sum of all scores
        # TRAIN: is given the RNN output matrix and the ground truth text and computes the loss value
        # INFER: given the RNN output matrix, decode the text
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3D, [1, 0, 2])

        with tf.compat.v1.name_scope('CTC_Loss'):
            # Compute the loss value for a batch
            self.groundTruthTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[  # Ground truth text as sparse tensor
                None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))
            self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(
                tf.nn.ctc_loss(labels=self.groundTruthTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                               ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))

        with tf.compat.v1.name_scope('CTC_Decoder'):
            # Decoder: Best path decoding
            # The simplest decoding
            # Takes the output of of the NN and computes an approximation by taking the most likely character at each position
            self.decoder = tf.compat.v1.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)

        # Return a CTC operation to compute the loss and CTC operation to decode the RNN output
        return self.loss, self.decoder

    def setupTF(self):
        """ Initialize TensorFlow """

        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        session = tf.compat.v1.Session()  # Tensorflow session
        saver = tf.compat.v1.train.Saver(max_to_keep=3)  # Saver is used to save a model to a file

        modelDirectory = './model/'  # directory where models are saved

        latestSnapshot = tf.train.latest_checkpoint(modelDirectory)  # Check if there is a saved model

        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(session, latestSnapshot)
        else:
            print('Init with new values')
            session.run(tf.compat.v1.global_variables_initializer())

        return session, saver

    def trainBatch(self, batch, batchNum):
        """
        Feed a batch into the NN to train it
        Return loss value for the batch
        """
        sparse = self.toSparse(batch.groundTruthTexts)  # ground truth texts as sparse tensor
        rate = 0.001  # learning rate
        evalList = [self.optimizer, self.loss]
        feedDictionary = {self.inputImages: batch.images, self.groundTruthTexts: sparse,
                    self.seqLen: [Model.maxTextLen] * Model.batchSize,
                    self.learningRate: rate}
        (_, lossValue) = self.session.run(evalList, feedDictionary)

        self.batchesTrained += 1
        return lossValue

    def inferBatch(self, batch):
        """
        Feed a batch into the NN to recognize texts
        Return the texts as decoded
        """
        numberOfElementsInBatch = len(batch.images)
        feedDictionary = {self.inputImages: batch.images, self.seqLen: [Model.maxTextLen] * numberOfElementsInBatch}
        evaluationResults = self.session.run([self.decoder, self.ctcIn3dTBC], feedDictionary)
        decoderOutput = evaluationResults[0]

        texts = self.decoderOutputToText(decoderOutput)
        return texts

    def toSparse(self, texts):
        """ Convert ground truth texts into sparse tensor for ctc_loss """
        indices = []
        values = []
        shape = [len(texts), 0]
        for (batchElement, texts) in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            # print(texts)
            labelStr = []
            for text in texts:
                # print(text, '|', end='')
                labelStr.append(self.charactersList.index(text))
            # print(' ')
            labelStr = [self.charactersList.index(c) for c in texts]
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoderOutputToText(self, ctcOutput):
        """ Extract texts from output of CTC decoder """
        # Contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(Model.batchSize)]

        # Ctc returns a tuple, first element is SparseTensor
        decoded = ctcOutput[0][0]
        # Go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]  # index according to [b,t]
            encodedLabelStrs[batchElement].append(label)
        # Map labels to chars for all batch elements
        return [str().join([self.charactersList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def save(self):
        """ Save a model to a file """
        self.snapshotID += 1
        self.saver.save(self.session, './model/snapshot',
                        global_step=self.snapshotID)
