from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import Counter
import shutil
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet
import sys

np.set_printoptions(threshold=np.nan)


def main(args):
    image_predictions = []

    with tf.Graph().as_default():
        sess = tf.Session()
        facenet.load_model(args.model_dir, sess)

        validation_set = facenet.get_dataset(args.data_dir)
        image_list, label_list = facenet.get_image_paths_and_labels(validation_set)

        # Run forward pass to calculate embeddings
        nrof_images = len(image_list)
        batch_size = args.batch_size
        nrof_batches = (nrof_images // batch_size) + 1

        for i in range(nrof_batches):
            if i == nrof_batches - 1:
                n = nrof_images
            else:
                n = i * batch_size + batch_size

            images = facenet.load_data(image_list[i * batch_size:n], False, False, args.image_size, do_prewhiten=True)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            logits = tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            scores = tf.nn.sigmoid(logits)

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            probs = sess.run(scores, feed_dict=feed_dict)
            print('Completed batch', i + 1, 'of', nrof_batches)

            name = image_list[i * batch_size:n]

            for j in range(len(probs)):
                indice = str(np.argmax(probs[j]))
                value = str(np.amax(probs[j]))

                image_predictions.append((str(name[j]), indice, value))

    accuracy = get_label(image_predictions, args.map_file)
    print("Classification accuracy: ", accuracy)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def get_label(preds, map_file):
    correct_counter, wrong_counter = 0, 0
    info = match_category_name(map_file)
    for flnm, indc, acc in preds:
        label_indice = map_labels(indc, map_file)
        label_indice = sorted(label_indice, key=lambda x: int(x[2]), reverse=True)

        if len(label_indice) == 0:
            print("The file: " + str(flnm) + " is unknown.")
            name = "unknown"
        elif len(label_indice) == 1:
            print("The file: " + str(flnm) + " is classified as: " + str(label_indice[0][0]))
            name = str(label_indice[0][0])
        else:
            print("The file: " + str(flnm) + " is classified as: " + str(label_indice[0][0]))
            name = str(label_indice[0][0])

        category = str(flnm).split("\t")[0].split("/")[-2]

        if (category, name) in info:
            correct_counter += 1
        elif category == "unknown" and name == "unknown":
            correct_counter += 1
        else:
            wrong_counter += 1

    return correct_counter / (correct_counter + wrong_counter)


def match_category_name(map_file_dir):
    map_file = open(map_file_dir)
    info = []

    for line in map_file:
        if line.startswith("m"):
            cat = line.split("\t")[0]
            name = line.split("\t")[1]
            info.append((cat, name))

    return info


def map_labels(indc, map_file):
    map_file = open(map_file)
    labels = []

    for line in map_file.read().splitlines():
        if line.startswith("m"):
            category = line.split("\t")[0]
            label = line.split("\t")[1]
            indice = line.split("\t")[2]
            file_counter = line.split("\t")[3]

            if indice == indc:
                labels.append((label, indice, file_counter))

    return labels


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing test images.')
    parser.add_argument('model_dir', type=str,
                        help='Path to the data directory containing the meta_file and ckpt_file.')
    parser.add_argument('map_file', type=str,
                        help='Path to the mapping file.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Size of input images.', default=160)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))