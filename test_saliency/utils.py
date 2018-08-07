import gzip
import numpy as np

BATCH_SIZE = 64

aaMap_fang = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
              'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'NoSeq': 20}
aaString_Fang = 'ARNDCQEGHILKMFPSTWYV_'

aaString_jurtz = 'ACEDGFIHKMLNQPSRTWVYX'
pssmString_jurtz = 'ACDEFGHIKLMNPQRSTVWXY'
aaMap_jurtz = {amino: i for i, amino in enumerate(aaString_jurtz)}
pssmMap_jurtz = {amino: i for i, amino in enumerate(pssmString_jurtz)}

ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
ssConvertString = 'CBEGIHST'


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

    def get_batch_from_seq(self, sequence):
        self.update_data(sequence)

        batch = sequence // BATCH_SIZE
        if self.subset is "valid":
            batch -= 5248 // BATCH_SIZE
        elif self.subset is "test":
            batch -= 5504 // BATCH_SIZE

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
        else:
            print("Problem with data managing")
            assert False
