import gzip
import numpy as np

BATCH_SIZE = 64
WINDOW = 9

aaMap_fang = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
              'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'NoSeq': 20}
aaString_Fang = 'ARNDCQEGHILKMFPSTWYV_'

aaString_jurtz = 'ACEDGFIHKMLNQPSRTWVYX'
pssmString_jurtz = 'ACDEFGHIKLMNPQRSTVWXY'
aaMap_jurtz = {amino: i for i, amino in enumerate(aaString_jurtz)}
pssmMap_jurtz = {amino: i for i, amino in enumerate(pssmString_jurtz)}

ssConvertString = 'LBEGIHST'


def toSeqLogo(total):
    print(" ".join(pssmString_jurtz))
    for row in range(len(total)):
        print(" ".join("{:.8f}".format(el) for el in total[row, 21:]))


def convertPredictQ8Result2HumanReadable(predictedSS):
    predSS = np.argmax(predictedSS, axis=-1)
    result = []
    for i in range(0, 700):
        result.append(ssConvertMap[predSS[i]])
    return ''.join(result)


def decode(coded_seq, map=aaMap_fang):
    coded_seq = np.argmax(coded_seq, axis=1)
    decoded_seq = []

    for number in coded_seq:
        for am, code in map.items():
            if code == number:  # and am is not 'NoSeq':
                decoded_seq.append(am)

    return "".join(decoded_seq)


def load_gz(path):  # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


class Jurtz_Data():
    TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
    TEST_PATH = 'cb513+profile_split1.npy.gz'

    def __init__(self):
        from data import get_train, get_test
        self.get_train = get_train
        self.get_test = get_test

        self.subset = None
        self.X = None
        self.mask = None
        self.split_value = None

    def get_batch_from_seq(self, sequence):
        self.update_data(sequence)

        batch = sequence // BATCH_SIZE
        if self.subset is "valid":
            batch -= 5248 // BATCH_SIZE
        elif self.subset is "test":
            batch -= 5504 // BATCH_SIZE
        elif self.subset is not "train":
            raise ValueError("Error: subset is " + str(self.subset))

        idx = range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        return self.X[idx], self.mask[idx]

    def get_sequence(self, sequence):
        self.update_data(sequence)

        if self.subset is "valid":
            sequence -= 5248
        elif self.subset is "test":
            sequence -= 5504

        return self.X[sequence], self.mask[sequence]

    def update_data(self, sequence):
        if sequence < 5248 and self.subset is not "train":
            self.X, _, _, _, self.mask, _, _ = self.get_train(Jurtz_Data.TRAIN_PATH)
            self.subset = "train"
        elif sequence >= 5248 and sequence < 5504 and self.subset is not "valid":
            _, self.X, _, _, _, self.mask, _ = self.get_train(Jurtz_Data.TRAIN_PATH)
            self.subset = "valid"
        elif sequence >= 5504 and self.subset is not "test":
            self.X, self.mask, _, _ = self.get_test(Jurtz_Data.TEST_PATH)
            self.subset = "test"

    def get_all_data(self):
        X_train, X_valid, labels_train, labels_valid, mask_train, mask_valid, _ = self.get_train(Jurtz_Data.TRAIN_PATH)
        X_test, mask_test, labels_test, _ = self.get_test(Jurtz_Data.TEST_PATH)
        # print X_train[:-30].shape, X_valid.shape, X_test[:-126].shape
        self.split_value = len(X_train[:-30]) + len(X_valid)
        # print split_value

        X = np.concatenate((X_train[:-30], X_valid, X_test[:-126]))
        labels = np.concatenate((labels_train[:-30], labels_valid, labels_test[:-126]))
        mask = np.concatenate((mask_train[:-30], mask_valid, mask_test[:-126]))
        # print X.shape

        return X, labels, mask

    def get_all_predictions(self):
        predictions_path = "../secondary_proteins_prediction/predictions/predictions_train_valid_pureConv-20180804-010835-47.npy"
        predictions = np.load(predictions_path)
        # print "train_val", predictions.shape

        predictions_path = "../secondary_proteins_prediction/predictions/predictionstest_pureConv-20180804-010835-47.npy"
        predictions2 = np.load(predictions_path)
        # print "test", predictions2[:-126].shape

        return np.concatenate((predictions, predictions2[:-126]))
