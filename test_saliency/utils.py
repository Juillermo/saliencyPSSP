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


def toFang(X, mask):
    # Permutation from Troyanska's pssm arranging (ACDEFGHIKLMNPQRSTVWXY) to Fang's ('ARNDCQEGHILKMFPSTWYV NoSeq')
    # Amino Acid 'X' was treated as amino acid 'A'. 'B' was treated as amino acid 'N'. 'Z' was treated as amino acid 'Q'
    sorted_fang = sorted(aaMap_fang.keys(), key=lambda letter: aaMap_fang[letter])

    aaMap = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    pssm = 'ACDEFGHIKLMNPQRSTVWXY'
    aaMap_jurtz = {amino: i for i, amino in enumerate(aaMap)}
    pssm_jurtz = {amino: i for i, amino in enumerate(pssm)}

    index_am = np.array([aaMap_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])
    index_pssm = np.array([pssm_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])

    X_am = X[:, :, index_am]
    X_pssm = X[:, :, index_pssm + 21]

    # Amino Acid 'X' was treated as amino acid 'A'. 'B' was treated as amino acid 'N'. 'Z' was treated as amino acid 'Q
    X_am[:, :, aaMap_fang['A']] = X[:, :, aaMap_jurtz['A']] + X[:, :, aaMap_jurtz['X']]
    X_pssm[:,:,aaMap_fang['A']] = X[:, :, pssm_jurtz['A']] + X[:,:, pssm_jurtz['X']]

    # Add NoSeq class
    mask_inv = mask * -1 + 1
    X_am = np.concatenate([X_am, mask_inv[:, :, None]], axis=2)
    X_pssm = np.concatenate([X_pssm, mask_inv[:, :, None]], axis=2)

    return X_am, X_pssm
