import glob
import os
import re
import argparse

import numpy as np

from saliency import BATCH_SIZE, compute_tensor_jurtz
from utils import ssConvertString, Jurtz_Data

def probe():
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
    return exists

def repair_saliencies(args):
    dater = Jurtz_Data()

    batch_range = range(6018 // BATCH_SIZE)
    if args.dir == 'b':
        batch_range = reversed(batch_range)
    elif args.dir != 'f':
        raise ValueError("args.dir is " + str(args.dir))

    for batch in batch_range:
        exists = probe()
        for batch_seq in range(BATCH_SIZE):
            seq = batch * BATCH_SIZE + batch_seq
            if int(np.sum(exists[seq])) != 8:
                X_batch, mask_batch = dater.get_batch_from_seq(seq)
                for label in ssConvertString:
                    if not exists[seq, ssConvertString.find(label)]:
                        print("Repairing sequence {:d} and batch {:d} for label {:s}".format(seq, batch, label))
                        compute_tensor_jurtz(X_batch, mask_batch, batch, label, ini=batch_seq)
                break


def assert_all():
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')
    files = glob.glob('saliencies*')

    exists = np.zeros((6018, 8))
    for el in files:
        found = re.search(r'(\d+)(\D)', el).groups()
        num = int(found[0])
        label = ssConvertString.find(found[1])
        exists[num, label] = 1

    os.chdir(origin)

    seq_rem = 0
    sal_rem = 0
    for i, el in enumerate(exists):
        tot = int(np.sum(el))
        if tot != 0 and tot != 8:
            seq_rem += 1
            sal_rem += 8 - tot
            print(i, tot)

    print(str(seq_rem) + " sequences remaining")
    print(str(sal_rem) + " saliencies remaining")


def main():
    parser = argparse.ArgumentParser(description='For seen saliency files and repairing them')
    parser.add_argument('--function', type=str, default='assert', metavar='function',
                        help='which function of the file to use (assert, repair)')
    parser.add_argument('--dir', type=str, default='f', metavar='dir',
                        help='direction of the reparation process (f or b)')
    args = parser.parse_args()

    if args.function == "assert":
        assert_all()
    elif args.function == "repair":
        repair_saliencies(args)
    else:
        print("No valid function selected")


if __name__ == "__main__":
    main()
    # probe()
    # repair_saliencies()
