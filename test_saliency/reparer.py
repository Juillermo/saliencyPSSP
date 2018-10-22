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
from utils import ssConvertString, Jurtz_Data, WINDOW

SCRATCH_PATH = '/scratch/grm1g17/'

SALIENCIES_SCRATCH_PATH = SCRATCH_PATH + 'saliencies/'
PROCESSED_SCRATCH_PATH = SCRATCH_PATH + 'processed/'
NUM_SEQS = 6016


def probe(folder='saliencies'):
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)
    files = glob.glob(folder + '*')

    exists = np.zeros((NUM_SEQS, 8))
    for el in files:
        found = re.search(r'(\d+)(\D)', el).groups()
        num = int(found[0])
        label = ssConvertString.find(found[1])
        if num < NUM_SEQS:
            assert exists[num, label] == 0
            exists[num, label] += 1

    os.chdir(origin)
    return exists


def scrap():
    exists = probe('saliencies')

    fail_seqs = 0
    deleted = 0
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)
    for seq in range(len(exists)):
        for label in ssConvertString:
            if exists[seq, ssConvertString.find(label)]:
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

                except OSError:
                    fail_seqs += 1
                    print(str(seq) + label + " Not found")

    os.chdir(origin)
    print(str(deleted) + " saliencies deleted")
    print(str(NUM_SEQS * 8 - fail_seqs) + " saliencies remaining")


def process():
    exists_sal = probe('saliencies')
    exists_proc = probe('processed')

    dater = Jurtz_Data()

    fail_seqs = 0
    processed = 0
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)
    for seq in range(len(exists_sal)):
        X_seq, mask_seq = dater.get_sequence(seq)
        end_seq = int(sum(mask_seq))
        processed_seq = np.zeros((end_seq, 8, 2 * WINDOW + 1, 42))

        if int(np.sum(exists_sal[seq])) != 8:
            print("Saliencies for " + str(seq) + " incomplete, " + str(np.sum(exists_sal[seq])) + " found")
        elif int(np.sum(exists_proc[seq])) != 8:
            try:
                for label in ssConvertString:
                    try:
                        fname = "saliencies" + str(seq) + label + ".pkl"
                        with open(fname, "rb") as f:
                            saliency_seq = np.array(pickle.load(f))
                    except OSError:
                        fname = "saliencies{:4d}{:s}.pkl".format(seq, label)
                        with open(fname, "rb") as f:
                            saliency_seq = np.array(pickle.load(f))

                    for pos in range(end_seq):
                        saliency_pos = np.zeros((2 * WINDOW + 1, 42))  # window-size, n aminoacids
                        # Pre-WINDOW
                        if pos > WINDOW:
                            init = pos - WINDOW
                            saliency_pos[:WINDOW] += np.multiply(saliency_seq[pos, init:pos, :], X_seq[init:pos])
                        elif pos != 0:
                            init = WINDOW - pos
                            saliency_pos[init:WINDOW] += np.multiply(saliency_seq[pos, 0:pos, :], X_seq[0:pos])

                        # Window
                        saliency_pos[WINDOW] += np.multiply(saliency_seq[pos, pos, :], X_seq[pos])

                        # Post-WINDOW
                        if pos + WINDOW + 1 <= end_seq:
                            end = pos + WINDOW + 1
                            saliency_pos[WINDOW + 1:] += np.multiply(saliency_seq[pos, pos + 1:end, :],
                                                                     X_seq[pos + 1:end])
                        elif pos != end_seq:
                            end = end_seq
                            saliency_pos[WINDOW + 1:-(pos + WINDOW + 1 - end)] += np.multiply(
                                saliency_seq[pos, pos + 1:end, :],
                                X_seq[pos + 1:end])

                        processed_seq[pos, ssConvertString.find(label)] = saliency_pos
                        processed += 1

            except OSError:
                fail_seqs += 1
                print(str(seq) + " Files not found")

        fname = "saliencies{:4d}.npy".format(seq)
        np.save(PROCESSED_SCRATCH_PATH + fname, processed_seq)

    os.chdir(origin)
    print(str(processed) + " saliencies processed")
    print(str(fail_seqs) + " saliencies failed")


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

    batch_range = range(NUM_SEQS // BATCH_SIZE)
    if args.dir == 'b':
        batch_range = reversed(batch_range)
    elif args.dir != 'f':
        raise ValueError("args.dir is " + str(args.dir))

    for batch in batch_range:
        exists = probe('saliencies')
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
    exists = probe('saliencies')

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
    parser.add_argument('--func', type=str, default='assert', metavar='func',
                        help='which function of the file to use (assert, repair, scrap, process)')
    parser.add_argument('--dir', type=str, default='f', metavar='dir',
                        help='direction of the reparation process (f or b)')
    args = parser.parse_args()

    if args.func == "assert":
        assert_all()
    elif args.func == "repair":
        repair_saliencies(args)
    elif args.func == "scrap":
        scrap()
    elif args.func == "process":
        process()
    else:
        print("No valid function selected")


if __name__ == "__main__":
    main()
