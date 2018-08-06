import glob
import os
import re

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

    print([str(i)+" "+el for i, el in exists])