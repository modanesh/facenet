from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import shutil
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet

np.set_printoptions(threshold=np.nan)

def get_prob_directory(images_dir, sess):
    image_predictions = []

    for filename in os.listdir(images_dir):

        # image = np.zeros((160, 160, 3))
        # img = Image.open(images_dir + filename)
        # img = img.resize((160, 160))
        # image = np.array(img).reshape(1, 160, 160, 3)

        img = misc.imread(images_dir + "/" + filename)
        if img.ndim == 2:
            img = to_rgb(img)
        img = prewhiten(img)
        image = np.array(img).reshape(1, 160, 160, 3)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        scores = tf.nn.sigmoid(logits)
        probs = sess.run(scores, feed_dict=feed_dict)

        name = str(filename)
        indice = str(np.argmax(probs))
        value = str(np.amax(probs))

        image_predictions.append((name, indice, value))

        print("hi hi hi")

        print(probs)
        print(len(probs))
        print(indice)
        print(value)


    return image_predictions

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


def clean_dataset(image_predictions, to_be_cleaned_data_dir):
    image_info = []
    indices = []

    for prediction in image_predictions:
        image_name = prediction[0]
        image_indice = prediction[1]
        image_confidence = prediction[2]

        indices.append(int(image_indice))
        image_info.append((image_name, int(image_indice), float(image_confidence)))

    most_common, num_most_common = Counter(indices).most_common(1)[0]

    for i in range(0, len(image_info)):
        if image_info[i][1] != most_common:
            # shutil.move((to_be_cleaned_data_dir + str(image_info[i][0])), (outlier_data_dir + str(image_info[i][0])))
            os.remove(to_be_cleaned_data_dir + str(image_info[i][0]))



if __name__ == '__main__':

    model_dir = "/Users/Mohamad/Sensifai/FaceNet/data/ms-celeb-1m/"
    base_dir = "/Users/Mohamad/Sensifai/FaceNet/handle_align/data/"
    clean_base_dir = "/Users/Mohamad/Sensifai/FaceNet/handle_align/cleaned 2/"
    # model_dir = "/home/deepface/users/danesh/FaceNet/data/ms-celeb-1m/"
    # base_dir = "/media/deepface/5a858105-5c78-47d2-b190-7fdf640e89b6/MS-Celeb-1M-Aligned-Faces/aligned_data/"
    # clean_base_dir = "/media/deepface/5a858105-5c78-47d2-b190-7fdf640e89b6/MS-Celeb-1M-Aligned-Faces/clean_aligned/"

    # visited_folders = ["m.072mhn", "m.04g_1z", "m.03f2wj1", "m.0kt5r2", "m.03w5_w", "m.03m4vl", "m.0pmf5c4",
    #                    "m.02w03b8", "m.02pjcb2", "m.063ynp", "m.02fj_x", "m.027r4pz", "m.025vvw5", "m.05fbd9p",
    #                    "m.02w_12w", "m.027nj_2", "m.05j7ln", "m.0gkztc9", "m.04h6s_", "m.02qtsh2", "m.0fp_5hq",
    #                    "m.0450ck", "m.0bhbr1q", "m.0d28m39", "m.021hxp", "m.025dnm", "m.0c0vl", "m.03mdtcp",
    #                    "m.0j28gmr", "m.02q041", "m.01wn0cq", "m.029y7m", "m.03d0pkw", "m.048z00", "m.027ldsh",
    #                    "m.01pk3z", "m.0gtbrg", "m.0qzgxpk", "m.0415v8p", "m.04jb7vq", "m.0879p9_", "m.0bqx7g",
    #                    "m.0gbznyf", "m.09cxd3", "m.04hpck", "m.09fdg1", "m.0hr3r3q", "m.07v79y", "m.02r51v1",
    #                    "m.05p6bss", "m.0265tb5", "m.01yhn5", "m.0b9jth", "m.01n0pn", "m.0ctlx6", "m.02z090b",
    #                    "m.0cks06", "m.0268x4h", "m.04wj2x", "m.0bxzygh", "m.0j582", "m.04q36md", "m.04kr63w",
    #                    "m.0gywlgf", "m.0r4mn33", "m.0c41jsn", "m.0g9rc7", "m.0b8h20", "m.04ctfn9", "m.03mk6l",
    #                    "m.0ndjn8f", "m.09nm4_", "m.06p83", "m.06zjmsg", "m.05lgzf", "m.07tcyd", "m.09k7hfp",
    #                    "m.0gtvt6x", "m.012v1t", "m.0gcv1y", "m.05c3s8k", "m.04cwdp7", "m.04z_59j", "m.06yv3c",
    #                    "m.03g_t2v", "m.03z8zh", "m.06k8bp", "m.01pc1lt", "m.09gh51b", "m.05zpzn_", "m.05zqz12",
    #                    "m.05tk7y", "m.0507ww", "m.020yg2", "m.0bhgvg", "m.03wz87", "m.0c5pv", "m.04y79_n",
    #                    "m.01r8wg", "m.024x6g", "m.015c4g", "m.027qtb9", "m.02qvqwr", "m.01qmj84", "m.0dhx0q",
    #                    "m.05hc81", "m.03y8dfy", "m.04f1s6", "m.0dlmq00", "m.0430kf", "m.03ysz4", "m.01gct2",
    #                    "m.0gjchn4", "m.09x8ms", "m.04gvt5", "m.02qhg7v", "m.0gffmw3", "m.0416h9j", "m.0b10q3",
    #                    "m.0dlk6dy", "m.06w9rdb", "m.0gfg4h9", "m.0gtvbgz", "m.0d3__v", "m.02rp4tl", "m.06_ydgx",
    #                    "m.0pyww", "m.05mgqb", "m.0gx0x90", "m.04sj7s", "m.02qj6m0", "m.0bf32y", "m.0gtsp_",
    #                    "m.0j3db92", "m.09gl3v", "m.0jhf3", "m.0f_v5", "m.03bx0zz", "m.072t_0", "m.084_42",
    #                    "m.0ktn3", "m.07kdybt", "m.0bwfzyw", "m.0zdsnx5", "m.0fpcbv", "m.02vvvf5", "m.0dmszb",
    #                    "m.03nn1x8", "m.02qfvrv", "m.03bptr", "m.0hgn1wh", "m.0466jjy", "m.0c3bd3", "m.0h_czy1",
    #                    "m.026k3jb", "m.01s8t4", "m.07k10c", "m.05wcm_", "m.05pc22d", "m.0jt7ypz", "m.0hzrsld",
    #                    "m.0gtkxhn", "m.02b9g4", "m.0gw7qp0", "m.07rcs3", "m.02plthk", "m.04cth0", "m.0rpjnv2",
    #                    "m.02x7s6t", "m.06cbp8", "m.0w2zycd", "m.0c3yr03", "m.02pzbyk", "m.0j29gh3", "m.0d3t0d",
    #                    "m.03gqd7m", "m.0h38bk", "m.04gv3w3", "m.0417kj1", "m.0d5sxj", "m.07kg187", "m.03fdvz",
    #                    "m.09rx8mq", "m.07kdpw_", "m.054l9v", "m.0vsg9hs", "m.0bh83vj", "m.0310_t", "m.0dljhhg",
    #                    "m.0bbzynh", "m.072dp_", "m.044k7x", "m.047p4wl", "m.0l020w", "m.03ck5mv"]

    visited_folders = []

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir, sess)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            logits = tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for foldername in os.listdir(clean_base_dir):
                if foldername not in visited_folders:
                    print(foldername)
                    predictions = get_prob_directory(clean_base_dir + foldername, sess)

                    to_be_cleaned_data_dir = clean_base_dir + foldername + "/"
                    clean_dataset(predictions, to_be_cleaned_data_dir)
