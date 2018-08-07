import glob
import os
import re
import argparse

import numpy as np

from saliency import BATCH_SIZE, compute_tensor_jurtz
from utils import ssConvertString, Jurtz_Data


def repair_saliencies(args):
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
    dater = Jurtz_Data()

    for batch in range(6018 // BATCH_SIZE):
        if args.dir == 'b':
            batch = 6018 // BATCH_SIZE - batch
        for batch_seq in range(BATCH_SIZE):
            seq = batch * BATCH_SIZE + batch_seq
            if int(np.sum(exists[seq])) != 8:
                X_batch, mask_batch = dater.get_batch_from_seq(seq)
                for label in ssConvertString:
                    if not exists[seq, ssConvertString.find(label)]:
                        compute_tensor_jurtz(X_batch, mask_batch, batch, label, ini=batch_seq)
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

    print([str(i) + " " + str(el) for i, el in enumerate(exists)])


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

    for i, el in enumerate(exists):
        tot = np.sum(el)
        if int(tot) != 0 and int(tot) != 8:
            print(i, tot)

def main():
    parser = argparse.ArgumentParser(description='For seen saliency files and repairing them')
    parser.add_argument('--function', type=str, default='probe', metavar='function',
                        help='which function of the file to use (probe, assert, repair)')
    parser.add_argument('--dir', type=str, default='f', metavar='dir',
                        help='direction of the reparation process (f or b)')
    args = parser.parse_args()

    if args.function == "probe":
        probe()
    elif args.function == "assert":
        assert_all()
    elif args.function == "repair":
        repair_saliencies(args)
    else:
        print("No valid function selected")

if __name__ == "__main__":
    main()
    #probe()
    #repair_saliencies()
