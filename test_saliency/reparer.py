import glob
import os
import re
import argparse
import importlib
import pickle

import numpy as np
import theano.tensor as T
import lasagne as nn

from saliency import BATCH_SIZE, compute_complex_saliency
from utils import ssConvertString, Jurtz_Data

SALIENCIES_SCRATCH_PATH = '/scratch/grm1g17/saliencies'

def probe():
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)
    files = glob.glob('saliencies*')

    exists = np.zeros((6018, 8))
    for el in files:
        found = re.search(r'(\d+)(\D)', el).groups()
        num = int(found[0])
        label = ssConvertString.find(found[1])
        if num < 6018:
            exists[num, label] += 1

    os.chdir(origin)
    return exists

def scrapper():
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)

    fail_seqs = 0
    deleted = 0
    for seq in range(6018):
        for label in ssConvertString:
            try:
                try:
                    fname = "saliencies" + str(seq) + label + ".pkl"
                    with open(fname, "rb") as f:
                        saliency = np.array(pickle.load(f))
                except OSError:
                    fname = "saliencies{:4d}{:s}.pkl".format(seq, label)
                    with open(fname, "rb") as f:
                        saliency = np.array(pickle.load(f))

                if saliency.ndim != 3 or saliency.shape[0] != saliency.shape[1]:
                    os.remove(fname)
                    print("File " + fname + " deleted")
                    deleted += 1
                    raise OSError("saliency badly formatted")

                print(seq)
            except OSError:
                fail_seqs += 1
                print(str(seq) + " Not found")

    os.chdir(origin)
    print(str(deleted)+" saliencies deleted")
    print(str(6018 * 8 - fail_seqs) + " saliencies remaining")


def repair_saliencies(args):
    dater = Jurtz_Data()

    metadata_path = "dump_pureConv-20180804-010835-47.pkl"
    metadata = np.load(metadata_path)
    config_name = metadata['config_name']
    config = importlib.import_module("%s" % config_name)
    print("Using configurations: '%s'" % config_name)
    l_in, l_out = config.build_model()

    sym_x = T.tensor3()
    inference = nn.layers.get_output(
        l_out, sym_x, deterministic=True)
    nn.layers.set_all_param_values(l_out, metadata['param_values'])

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
                        compute_complex_saliency(X_batch=X_batch, mask_batch=mask_batch, batch=batch, label=label,
                                                 batch_seq=batch_seq, inference=inference, sym_x=sym_x)


def assert_all():
    exists = probe()

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
    elif args.function == "scrap":
        scrapper()
    else:
        print("No valid function selected")


if __name__ == "__main__":
    main()
    # probe()
    # repair_saliencies()
