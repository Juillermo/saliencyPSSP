import glob
import os
import re

import numpy as np

from saliency import batch_size, compute_tensor_jurtz
from utils import ssConvertString

def repare_saliencies():
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')
    files = glob.glob('saliencies*')
    exists = [False for _ in range(6018)]
    for el in files:
        num = int(re.search(r'\d+', el).group(0))
        exists[num] = True

    os.chdir(origin)
    from data import get_train, get_test
    TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
    TEST_PATH = 'cb513+profile_split1.npy.gz'
    X, _, _, _, mask, _, _ = get_train(TRAIN_PATH)

    def get_index(batch):
        return batch

    for batch in range(6018 // batch_size):
        if batch == 5278 // batch_size:
            _, X, _, _, _, mask, _ = get_train(TRAIN_PATH)

            def get_index(batch):
                return batch - 5278 // batch_size

        elif batch == 5534 // batch_size:
            X, mask, _, _ = get_test(TEST_PATH)

            def get_index(batch):
                return batch - 5534 // batch_size

        for batch_seq in range(batch_size):
            seq = batch * batch_size + batch_seq
            if not exists[seq]:
                idx = get_index(batch)
                for label in ssConvertString:
                    compute_tensor_jurtz(X[idx], mask[idx], batch, label, ini=batch_seq)
                break

def probe():
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')
    files = glob.glob('saliencies*')
    exists = [False for _ in range(6018)]
    for el in files:
        num = int(re.search(r'\d+', el).group(0))
        exists[num] = True

    os.chdir(origin)

    print([str(i)+" "+str(el) for i, el in enumerate(exists)])


def assert_all():
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')
    files = glob.glob('saliencies*')
    exists = np.zeros((6018, 8))
    for el in files:
        found = re.search(r'(\d+)(\D)', el).groups()
        num = int(found[0])
        label = ssConvertString.find(found[1])
        exists[num, label] += 1

    os.chdir(origin)

    for el in exists:
        tot = np.sum(el)
        assert(tot == 0 | tot == 8)

if __name__ == "__main__":
    probe()