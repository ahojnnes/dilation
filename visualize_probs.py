import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    probs = np.load(args.input_path)["prob"]

    probs_sum = np.sum(np.abs(probs), axis=2)
    probs /= probs_sum[:, :, None]

    for i in range(probs.shape[2]):
        plt.imshow(probs[:, :, i])
        # plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(args.output_path + "-{:03d}.png".format(i))
        plt.close()


if __name__ == "__main__":
    main()
