from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import tensorflow as tf
import numpy as np
import os
import facenet
import time

np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':

    start_time = time.time()

    model_dir = "/Users/Mohamad/Sensifai/FaceNet/data/ms-celeb-1m/"
    data_dir = "/Users/Mohamad/Sensifai/FaceNet/handle_align/cleaned/"
    # model_dir = "/home/deepface/users/danesh/FaceNet/data/ms-celeb-1m/"
    # data_dir = "/media/deepface/5a858105-5c78-47d2-b190-7fdf640e89b6/MS-Celeb-1M-Aligned-Faces/clean_aligned/"
    image_batch = 20
    image_size = 160
    indices_values = []
    image_predictions = []
    indices = []
    folders = []


    with tf.Graph().as_default():
        sess = tf.Session()
        facenet.load_model(model_dir, sess)

        train_set = facenet.get_dataset(data_dir)
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)

        # Run forward pass to calculate embeddings
        nrof_images = len(image_list)
        batch_size = image_batch
        nrof_batches = (nrof_images // batch_size) + 1

        for i in range(nrof_batches):
            if i == nrof_batches - 1:
                n = nrof_images
            else:
                n = i * batch_size + batch_size

            images = facenet.load_data(image_list[i * batch_size:n], False, False, image_size, do_prewhiten=True)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            logits = tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            scores = tf.nn.sigmoid(logits)
            probs = sess.run(scores, feed_dict=feed_dict)
            print('Completed batch', i + 1, 'of', nrof_batches)

            for i in range(len(probs)):
                indices_values.append((str(np.argmax(probs[i])), str(np.amax(probs[i]))))

    for i in range(len(image_list)):
        image_predictions.append((image_list[i].split("/")[-2], image_list[i].split("/")[-1], indices_values[i][0], indices_values[i][1]))
        folders.append(image_list[i].split("/")[-2])

    folders = list(set(folders))

    for folder in folders:
        indices = []
        single_category = []
        for category, filename, index, confidence in image_predictions:
            if folder == category:
                single_category.append((category, filename, index, confidence))
                indices.append(int(index))

        most_common, num_most_common = Counter(indices).most_common(1)[0]

        for i in range(len(single_category)):
            if int(single_category[i][2]) != most_common:
                os.remove(data_dir + single_category[i][0] + "/" + single_category[i][1])
                print(single_category[i][0], single_category[i][1])

    # print(image_predictions)
    print(time.time() - start_time)