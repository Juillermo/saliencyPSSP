import os
import pickle
import argparse

import numpy as np

from reparer import SALIENCIES_SCRATCH_PATH, PROCESSED_SCRATCH_PATH
from utils import Jurtz_Data, ssConvertString, WINDOW

SHEER_PATH = "sheer_data/"


def calculate_aa_pssm(args):
    origin = os.getcwd()
    os.chdir(PROCESSED_SCRATCH_PATH)

    fail_seqs = 0
    points = []
    for seq in range(args.num_seqs):
        try:
            fname = "saliencies{:4d}.npy".format(seq)
            processed_seq = np.load(fname)

            for saliency_map in processed_seq:
                points.append((np.sum(abs(saliency_map[..., :21])), np.sum(abs(saliency_map[..., 21:]))))
            print(seq)

        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    points = np.array(points)
    print(points.shape)

    os.chdir(origin)
    success_seqs = args.num_seqs - fail_seqs
    print("aa/pssm analysis of " + str(success_seqs) + " elements")
    np.save(SHEER_PATH + "aa_pssm" + str(success_seqs) + ".npy", points)


def calculate_sheer(args):
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)

    dater = Jurtz_Data()

    fail_seqs = 0
    total = np.zeros((2 * WINDOW + 1, 42))  # WINDOW-size, n aminoacids + pssm
    for seq in range(args.num_seqs):
        try:
            try:
                fname = "saliencies" + str(seq) + args.label + ".pkl"
                with open(fname, "rb") as f:
                    saliency = np.array(pickle.load(f))
            except OSError:
                fname = "saliencies{:4d}{:s}.pkl".format(seq, args.label)
                with open(fname, "rb") as f:
                    saliency = np.array(pickle.load(f))

            if saliency.ndim != 3 or saliency.shape[0] != saliency.shape[1]:
                os.remove(fname)
                print("File " + fname + " deleted")
                raise OSError("saliency badly formatted")

            X_seq, mask_seq = dater.get_sequence(seq)
            end_seq = int(sum(mask_seq))
            for pos in range(end_seq):
                # Pre-WINDOW
                if pos > WINDOW:
                    init = pos - WINDOW
                    total[:WINDOW] += np.multiply(saliency[pos, init:pos, :], X_seq[init:pos])
                elif pos != 0:
                    init = WINDOW - pos
                    total[init:WINDOW] += np.multiply(saliency[pos, 0:pos, :], X_seq[0:pos])

                # Window
                total[WINDOW] += np.multiply(saliency[pos, pos, :], X_seq[pos])

                # Post-WINDOW
                if pos + WINDOW + 1 <= end_seq:
                    end = pos + WINDOW + 1
                    total[WINDOW + 1:] += np.multiply(saliency[pos, pos + 1:end, :], X_seq[pos + 1:end])
                elif pos != end_seq:
                    end = end_seq
                    total[WINDOW + 1:-(pos + WINDOW + 1 - end)] += np.multiply(saliency[pos, pos + 1:end, :],
                                                                               X_seq[pos + 1:end])
            print(seq)
        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)
    print("Sheer addition of " + str(args.num_seqs - fail_seqs) + " elements")
    with open("sheer_data/sheer" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(total, f, protocol=2)


def calculate_sheer_abs(args):
    origin = os.getcwd()
    os.chdir(PROCESSED_SCRATCH_PATH)

    fail_seqs = 0
    total = np.zeros((8, 2 * WINDOW + 1, 42))  # classes, WINDOW-size, aminoacids+pssm
    for seq in range(args.num_seqs):
        try:
            fname = "saliencies{:4d}.npy".format(seq)
            processed_seq = np.load(fname)

            for saliency_map in processed_seq:
                total += abs(saliency_map)
            print(seq)

        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)
    success_seqs = args.num_seqs - fail_seqs
    print("Sheer addition of " + str(success_seqs) + " elements, absolute value")
    np.save(SHEER_PATH + "sheer" + str(success_seqs) + ".npy", total)


def calculate_points(args):
    origin = os.getcwd()
    os.chdir(PROCESSED_SCRATCH_PATH)

    fail_seqs = 0
    points = []
    for seq in range(args.num_seqs):
        try:
            fname = "saliencies{:4d}.npy".format(seq)
            processed_seq = np.load(fname)

            for processsed_pos in processed_seq:
                points.append(np.sum(processsed_pos[..., 21:], axis=1))

            print(seq)
        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)
    success_seqs = args.num_seqs - fail_seqs
    print("Clustering points of " + str(success_seqs) + " elements")
    np.save(SHEER_PATH + "points" + str(success_seqs) + ".npy", np.array(points))
    print(len(points))
    print(points[0].shape)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate saliencies either by sheer addition, compute points for clustering, or aggregate points as aa/pssm')

    parser.add_argument('--func', choices=['sheer', 'points', 'aapssm', 'sheerabs'],
                        help='Function: "sheer" addition of saliencies, "points" for clustering, "aampssm" values, "sheerabs" addition')
    parser.add_argument('--label', default='H', choices=[el for el in ssConvertString],
                        help='class from which to analyse the saliencies (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of sequences aggregated for SeqLogo (default 2)')
    args = parser.parse_args()

    if args.func == 'sheer':
        calculate_sheer(args)
    elif args.func == 'points':
        calculate_points(args)
    elif args.func == 'aapssm':
        calculate_aa_pssm(args)
    elif args.func == 'sheerabs':
        calculate_sheer_abs(args)
    else:
        raise ValueError('Function "' + args.func + '" not recognized, try with "sheer", "points", or "aapssm"')


if __name__ == "__main__":
    main()
