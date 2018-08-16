import pickle

import numpy as np
import matplotlib.pyplot as plt

from utils import WINDOW, ssConvertString, pssmString_jurtz

FIGURES_PATH = "../thesis/Figures/"

CLASS_COLOURS = {'L': 'blue', 'S': 'slateblue',
                 'E': 'red', 'T': 'coral', 'B': 'lightsalmon',
                 'H': 'green', 'G': 'limegreen', 'I': 'lightgreen'}

first = [int("E6", 16) / 255.0, 0, 0]
second = [0, int("D9", 16) / 255.0, 0]
third = [0, 0, int("FF", 16) / 255.0]
fourth = [0, 0, 0]
SEQLOGO_COLOURS = {'D': first, 'E': first,
                   'N': second, 'Q': second, 'S': second, 'G': second, 'T': second, 'Y': second,
                   'R': third, 'K': third, 'H': third,
                   'X': fourth, 'I': fourth, 'A': fourth, 'V': fourth, 'L': fourth, 'F': fourth}


def toSeqLogo(total):
    for row in range(len(total)):
        print(" ".join("{:.8f}".format(el) for el in total[row, 21:]))


def plot_aa_pssm():
    num_seqs = 6018
    with open("aa_pssm" + str(num_seqs) + ".pkl", "rb") as f:
        aa_pssm = pickle.load(f)
        print("Total shape", aa_pssm.shape)

    aa_tot = aa_pssm[:, 0]
    pssm_tot = aa_pssm[:, 1]
    superiority = np.sum(pssm_tot) / np.sum(aa_tot)
    print(superiority)

    for i in range(len(aa_tot)):
        if aa_tot[i] == 0 and pssm_tot[i] == 0:
            aa_tot[i] = 0.01
            pssm_tot[i] = 0.01

    # plt.plot(aa_tot == 0)
    # plt.plot(pssm_tot == 0)

    base_diff = (pssm_tot - aa_tot)  # / superiority

    # from matplotlib.ticker import PercentFormatter

    fig, (ax2, ax) = plt.subplots(1, 2, figsize=(12, 3))
    min_x, max_x = 1, 6
    max_y = 1.1
    xlim = [min_x, max_x]
    ylim = [0, max_y]
    bins = np.linspace(min_x, max_x, 2 * (max_x - min_x) + 1)

    diff1 = pssm_tot / aa_tot
    ax.hist(diff1[base_diff > 0], bins=bins, density=True)  # , log=True)
    ax.set(xlim=xlim, title="superior pssm saliency scores: {:.1e}".format(len(diff1[base_diff > 0])),
           ylabel="fraction of total",
           xlabel="times superior", ylim=ylim)
    ax.yaxis.set(ticks=np.linspace(0, 1, 6), ticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

    diff2 = aa_tot / pssm_tot
    ax2.hist(diff2[base_diff < 0], bins=bins, density=True)  # , log=True)
    ax2.set(xlim=xlim, title="superior amino-acid saliency scores: {:.1e}".format(len(diff2[base_diff < 0])),
            ylabel="fraction of total",
            xlabel="times superior", ylim=ylim)
    ax2.yaxis.set(ticks=np.linspace(0, 1, 6), ticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.tight_layout()
    # plt.savefig(FIGURES_PATH + "aa_pssm.eps", format="eps")
    plt.show()
    # plt.savefig("../thesis/Figures/aa_pssm.eps", format="eps")

    return base_diff, pssm_tot, aa_tot


def collect_saliencies(folder="processed/"):
    num_seqs = 1
    saliencies = []

    for seq in range(num_seqs):
        if folder == "processed/":
            fname = folder + "saliencies" + str(seq) + ".npy"
            saliencies = np.load(fname)
        else:
            for target_class in ssConvertString:
                with open(folder + "saliencies" + str(seq) + target_class + ".pkl", "rb") as f:
                    saliency = np.array(pickle.load(f))
                    saliencies.append(saliency)
                saliencies = np.array(saliencies)

    print("Saliency sequence shape", saliencies.shape)
    return saliencies


def plot_sliding_saliencies():
    saliencies = collect_saliencies()

    def plot_singles(saliencies):
        ini, end = 2, 8
        plot_len = end - ini
        fig, axes = plt.subplots(plot_len / 2, 2, figsize=(15, plot_len))
        for pos in range(ini, end):
            ax1 = axes[(pos - ini) % 3][(pos - ini) // 3]
            tot_plot = saliencies[pos, 5]

            ax1.plot(tot_plot[:, :])
            ax1.set(title="saliency map of position " + str(pos))
            ax1.xaxis.set(ticks=range(19), ticklabels=range(-9, 10))

        plt.tight_layout()
        # plt.savefig(FIGURES_PATH+"sliding.eps", format="eps")
        plt.show()

    def plot_for_clustering(saliencies):
        end_seq = len(saliencies)
        ini, end = 2, 8
        plot_len = end - ini
        fig, axes = plt.subplots(plot_len, 2, figsize=(15, 2 * plot_len))
        for pos in range(ini, end):
            (ax1, ax2) = (axes[pos - ini][0], axes[pos - ini][1])
            tot_plot = saliencies[pos, 5]

            ax1.plot(tot_plot[:, 21:])
            ax1.xaxis.set(ticks=range(19), ticklabels=range(-9, 10))

            ax2.plot(np.sum(tot_plot[:, 21:], axis=0))
            ax2.xaxis.set(ticks=range(21), ticklabels=pssmString_jurtz)
        plt.show()

    plot_singles(saliencies)


def plot_single_sequence():
    saliencies = collect_saliencies()
    preds = "EEEEEEEEELGHHHGLGTGLLSEEE"
    labels = "SEEEEEEEELHHHHHSGGGGSSLEE"
    aas = "IXAGVEVQLTDDFFADEKSISENYV"

    final_pos = 119
    ini_pos = 95
    end = final_pos + WINDOW + 1
    ini = ini_pos - WINDOW

    tot_sal = np.zeros((8, end - ini, 42))
    for pos in range(ini_pos, final_pos):
        # print(saliencies[pos].shape)
        # beg = ini_pos - ini
        # fin = beg + 2 * WINDOW + 1
        # print(beg, fin, tot_sal[:, beg:fin, :].shape, tot_sal.shape)

        tot_sal[:, (pos - ini_pos):(pos - ini_pos + 2 * WINDOW + 1)] += saliencies[pos]

    def plot_heatmap():
        fig, ax = plt.subplots(figsize=(7, 6.65))
        vmax = np.max(abs(tot_sal[5, :, 21:]))
        cax = ax.imshow(tot_sal[5, WINDOW:-WINDOW, 21:].T, cmap='PiYG', vmin=-vmax, vmax=vmax)
        # fig.colorbar(cax)

        ax.yaxis.set(ticks=range(len(pssmString_jurtz)), ticklabels=pssmString_jurtz)
        ax.xaxis.set(ticks=range(final_pos - ini_pos + 1),
                     ticklabels=[preds[i] + "\n" + el + "\n" + str(ini_pos + i) for i, el in
                                 enumerate(labels)])
        ax.margins(0)
        ax2 = ax.twiny()
        ax2.xaxis.set(ticks=range(final_pos - ini_pos + 1), ticklabels=aas)

        plt.tight_layout()
        # fig.savefig(FIGURES_PATH + "sample_Hclass.eps", format='eps')
        plt.show()

        toSeqLogo(tot_sal[5])

    plot_heatmap()

    tot_sal = np.sum(tot_sal, axis=2)

    def plot_lines():
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for j, label in enumerate("HGIETBLS"):
            ax.plot(tot_sal[ssConvertString.find(label), WINDOW:-WINDOW].T, marker='.', label=label,
                    color=CLASS_COLOURS[label])

        ax.legend(loc="upper left")

        ax.xaxis.set(ticks=range(final_pos - ini_pos + 1),
                     ticklabels=[preds[i] + "\n" + el + "\n" + str(ini_pos + i) for i, el in
                                 enumerate(labels)])
        colors = [CLASS_COLOURS[el] for el in preds]
        for color, tick in zip(colors, ax.xaxis.get_major_ticks()):
            tick.label1.set_color(color)  # set the color property

        ax.margins(0)
        ax2 = ax.twiny()
        ax2.xaxis.set(ticks=range(final_pos - ini_pos + 1), ticklabels=aas)
        # colors = [SEQLOGO_COLOURS[el] for el in aas]
        # for color, tick in zip(colors, ax2.xaxis.get_major_ticks()):
        # tick.label2.set_color(color)  # set the color property

        plt.tight_layout()
        fig.savefig(FIGURES_PATH + "sample_8classes.eps", format='eps')
        plt.show()

    # plot_lines()


# b, p, a = plot_aa_pssm()
# sal = collect_saliencies()
# plot_sliding_saliencies()
plot_single_sequence()

if __name__ == "__main__":
    # main()
    ""
