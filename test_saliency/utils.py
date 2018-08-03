import gzip
import numpy as np

aaMap_fang = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
              'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'NoSeq': 20}
aaString_Fang = 'ARNDCQEGHILKMFPSTWYV_'

ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
ssConvertString = 'CBEGIHST'

window = 29  # 29 aminoacids per side


def convertPredictQ8Result2HumanReadable(predictedSS):
    predSS = np.argmax(predictedSS, axis=-1)
    result = []
    for i in range(0, 700):
        result.append(ssConvertMap[predSS[i]])
    return ''.join(result)


def decode(coded_seq):
    coded_seq = np.argmax(coded_seq, axis=1)
    decoded_seq = []

    for number in coded_seq:
        for am, code in aaMap_fang.items():
            if code == number:  # and am is not 'NoSeq':
                decoded_seq.append(am)

    return "".join(decoded_seq)


def load_gz(path):  # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)
