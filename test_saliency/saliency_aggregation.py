import os
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from reparer import SALIENCIES_PATH, PROCESSED_PATH
from utils import Jurtz_Data, ssConvertString, WINDOW

SHEER_PATH = "sheer_data/"


def calculate_aa_pssm(args):
    origin = os.getcwd()
    os.chdir(PROCESSED_PATH)

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
    os.chdir(SALIENCIES_PATH)

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
    if args.func == "sheerabsequal":
        dater = Jurtz_Data()
        _, labels, mask = dater.get_all_data()
        predictions = dater.get_all_predictions()
        hits = np.zeros(len(ssConvertString))

    origin = os.getcwd()
    os.chdir(PROCESSED_PATH)

    fail_seqs = 0
    total = np.zeros((8, 2 * WINDOW + 1, 42))  # classes, WINDOW-size, aminoacids+pssm
    for seq in range(args.num_seqs):
        try:
            fname = "saliencies{:4d}.npy".format(seq)
            processed_seq = np.load(fname)

            for pos, saliency_map in enumerate(processed_seq):
                if args.func == "sheerabs":
                    total += abs(saliency_map)
                elif args.func == "sheerabsequal":
                    if mask[seq, pos]:
                        if labels[seq, pos] == np.argmax(predictions[seq, pos]):
                            total[labels[seq, pos]] += saliency_map[labels[seq, pos]]
                            hits[labels[seq, pos]] += 1
                    else:
                        break
                else:
                    raise ValueError(
                        "Invalid function '%s'. It should be either 'sheerabs' or 'sheerabsequal'.".format(args.func))
            print(seq)

        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)
    success_seqs = args.num_seqs - fail_seqs
    print("Sheer addition of " + str(success_seqs) + " elements, absolute value")
    np.save(SHEER_PATH + args.func + str(success_seqs) + ".npy", total)


def calculate_points(args):
    origin = os.getcwd()
    os.chdir(PROCESSED_PATH)

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
    # 1259916


def clustering(args):
    points = np.load(SHEER_PATH + "points" + str(args.num_seqs) + ".npy")
    points = points[:args.num_points, ssConvertString.find(args.label)]
    print("Original points shape:", points.shape)

    mask = np.ones(len(points), dtype='bool')
    for i, point in enumerate(points):
        if np.allclose(point, np.zeros_like(point)):
            mask[i] = False
    points = points[mask]
    print("Filtered points shape (no zero vectors):", points.shape)

    if args.clustering == "agglomerative":
        n_clusters = 4
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity="cosine")
    elif args.clustering == "DBSCAN":
        # model = DBSCAN(metric="cosine", eps=args.eps)
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=10)
    else:
        raise ValueError("Clustering algorithm '{:s}' not recognized".format(args.clustering))

    model.fit(points)

    file_end = "{:s}{:d}.npy".format(args.label, args.num_points)
    np.save(SHEER_PATH + args.clustering + file_end, model.labels_)
    np.save(SHEER_PATH + 'mask' + file_end, mask)
    print("Clustering completed. Labels saved in " + SHEER_PATH + args.clustering + file_end)
    print("Labels:")
    print(pd.Series(model.labels_).value_counts())


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate saliencies either by sheer addition, compute points for clustering, or aggregate points as aa/pssm')

    parser.add_argument('--func', choices=['sheer', 'points', 'aapssm', 'sheerabs', 'sheerabsequal', 'cluster'],
                        help='Function: "sheer" addition of saliencies, "points" for clustering, "aampssm" values, "sheerabs" addition, "sheerabsequal" addition only for predicted classes, "cluster" points')
    parser.add_argument('--label', default='H', choices=[el for el in ssConvertString],
                        help='class from which to analyse the saliencies (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of protein sequences aggregated (default 2)')
    parser.add_argument('--num-points', type=int, default=50, metavar='num_points',
                        help='number of points for clustering (default 50)')
    parser.add_argument('--clustering', choices=['agglomerative', 'DBSCAN'], default='agglomerative',
                        metavar='clustering', help='clustering algorithm being used (default agglomerative')
    parser.add_argument('--eps', type=float, default=0.5,
                        metavar='eps', help='epsilon for DBSCAN (default 0.5')
    args = parser.parse_args()

    if args.func == 'sheer':
        calculate_sheer(args)
    elif args.func == 'points':
        calculate_points(args)
    elif args.func == 'aapssm':
        calculate_aa_pssm(args)
    elif args.func == 'sheerabs' or 'sheerabsequal':
        calculate_sheer_abs(args)
    elif args.func == 'cluster':
        clustering(args)
    else:
        raise ValueError('Function "' + args.func + '" not recognized, try with "sheer", "points", or "aapssm"')


if __name__ == "__main__":
    main()
