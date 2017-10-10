from __future__ import print_function, division
import argparse
import caffe
import cv2
import os
import glob
import json
import multiprocessing
import numba
import numpy as np
from os.path import dirname, exists, join, splitext
import sys

import util


DATASETS = ["pascal_voc", "camvid", "kitti", "cityscapes"]
SUFFIXES = DATASETS + ["_overlay", "_disparity"]


class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(__file__)
        info_path = join(self.work_dir, "datasets", dataset_name + ".json")
        if not exists(info_path):
            raise IOError("Do not have information for dataset {}"
                          .format(dataset_name))
        with open(info_path, "r") as fp:
            info = json.load(fp)
        self.palette = np.array(info["palette"], dtype=np.uint8)
        self.mean_pixel = np.array(info["mean"], dtype=np.float32)
        self.dilation = info["dilation"]
        self.zoom = info["zoom"]
        self.name = dataset_name
        self.model_name = "dilation{}_{}".format(self.dilation, self.name)
        self.model_path = join(self.work_dir, "models",
                               self.model_name + "_deploy.prototxt")

    @property
    def pretrained_path(self):
        p = join(dirname(__file__), "pretrained",
                 self.model_name + ".caffemodel")
        if not exists(p):
            download_path = join(self.work_dir, "pretrained",
                                 "download_{}.sh".format(self.name))
            raise IOError("Pleaes run {} to download the pretrained network "
                          "weights first".format(download_path))
        return p




class PredictorState(object):

    def __init__(self, dataset_name):
        self.dataset = Dataset(dataset_name)
        self.net = caffe.Net(self.dataset.model_path,
                             self.dataset.pretrained_path,
                             caffe.TEST)
        self.input_dims = self.net.blobs["data"].shape
        self.caffe_in = np.zeros(self.input_dims, dtype=np.float32)


def rotate_image(image, n):
    if n == 0:
        return image
    image = np.atleast_3d(image)
    rotated_image = np.empty((image.shape[1], image.shape[0],
                              image.shape[2]), dtype=image.dtype)
    for d in range(image.shape[2]):
        rotated_image[..., d] = np.rot90(image[..., d], n)
    return rotated_image


predictor_state = None


def predict(dataset_name, input_path, output_path, rot90=0):
    global predictor_state
    if predictor_state is None:
        predictor_state = PredictorState(dataset_name)

    label_margin = 186
    batch_size, num_channels, input_height, input_width = predictor_state.input_dims

    image = cv2.imread(input_path, 1).astype(np.float32) - predictor_state.dataset.mean_pixel

    if image is None:
        return

    image = rotate_image(image, rot90)

    image_size = image.shape
    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin
    image = cv2.copyMakeBorder(image, label_margin, label_margin,
                               label_margin, label_margin,
                               cv2.BORDER_REFLECT_101)

    num_tiles_h = image_size[0] // output_height + \
                  (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + \
                  (1 if image_size[1] % output_width else 0)

    prediction = []
    for h in range(num_tiles_h):
        col_prediction = []
        for w in range(num_tiles_w):
            offset = [output_height * h,
                      output_width * w]
            tile = image[offset[0]:offset[0] + input_height,
                         offset[1]:offset[1] + input_width, :]
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)
            predictor_state.caffe_in[0] = tile.transpose([2, 0, 1])
            out = predictor_state.net.forward_all(**{predictor_state.net.inputs[0]: predictor_state.caffe_in})
            prob = out["prob"][0]
            col_prediction.append(prob)
        # print("concat row")
        col_prediction = np.concatenate(col_prediction, axis=2)
        prediction.append(col_prediction)

    prob = np.concatenate(prediction, axis=1)
    prob = util.interp_map(prob, predictor_state.dataset.zoom, image_size[1], image_size[0])
    prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)

    color_image = predictor_state.dataset.palette[prediction.ravel()].reshape(image_size)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    color_image = rotate_image(color_image, -rot90)

    print("Writing", output_path)
    cv2.imwrite(output_path, color_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--input_globs", required=True)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU ID to run CAFFE. "
                             "If -1 (default), CPU is used")
    parser.add_argument("--rot90", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        print("Using GPU ", args.gpu)
    else:
        caffe.set_mode_cpu()
        print("Using CPU")

    args.input_globs = args.input_globs.replace("\*", "*")

    for input_glob in args.input_globs.split():
        input_paths = sorted(glob.glob(input_glob))
        for input_path in input_paths:
            is_result = False
            for suffix in SUFFIXES:
                if input_path.endswith("%s.png" % suffix):
                    is_result = True
            if is_result:
                continue
            output_path = "{}_{}.png".format(splitext(input_path)[0], args.dataset)
            print("Processing", input_path)
            predict(args.dataset, input_path, output_path, args.rot90)


if __name__ == '__main__':
    main()
