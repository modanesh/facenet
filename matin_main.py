import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf
sys.path.insert(0, 'facenet/src/align/')
from align import detect_face
sys.path.insert(0, 'facenet/src/')
import classifier
import facenet

margin = 0
minsize = 30
factor = 0.709
image_size = 160
batch_size = 20
out_vid_fps = 29.0
frames_to_skip = 10
threshold = [0.6, 0.7, 0.7]
model_dir = '/Users/Mohamad/Sensifai/FaceNet/data/ms-celeb-1m/'
classifier_filename = '/Users/Mohamad/Sensifai/FaceNet/classifiers/clsfr_celeb20.neg3000.pkl'


def main():

    pnet_fun, rnet_fun, onet_fun = initMTCNN()
    sess, images_ph, embs, phase_train_ph, emb_size, model, class_names = loadModel()
    for file in os.listdir("/Users/Mohamad/Sensifai/FaceNet/data/aligned_images/Unknown/"):

        # pass input image here:
        image = cv2.imread("/Users/Mohamad/Sensifai/FaceNet/data/aligned_images/Unknown/" + file)

        # detect faces in given image and return bounding boxes
        boxes, faces = detect_faces_image(image, pnet_fun, rnet_fun, onet_fun)

        # recognize faces in given image and return bounding boxes, recognized labels and recognition probabilities
        boxes, faces, labels, probs = recognize_faces_image(image, pnet_fun, rnet_fun, onet_fun, sess, images_ph, embs, phase_train_ph, emb_size, model, class_names)

        # print results just for testing phase
        print('total of ' + str(len(boxes)) + ' faces detected and classified:')
        for label in labels:
            print(label)


def detect_faces_image(image, pnet_fun, rnet_fun, onet_fun):
    faces = []
    start_time = time.time()
    boxes, points = detect_face.detect_face(fixDim(image), minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    print("_____________________")
    print(time.time() - start_time)

    for b in boxes:
        x1, y1, x2, y2 = max(0, min(int(b[0]), image.shape[1])), max(0, min(int(b[1]), image.shape[0])), max(0, min(int(b[2]), image.shape[1])), max(0, min(int(b[3]), image.shape[0]))
        faces.append(image[y1:y2, x1:x2, :])
    return boxes, faces


def recognize_faces_image(image, pnet_fun, rnet_fun, onet_fun, sess, images_ph, embs, phase_train_ph, emb_size, model, class_names):
    faces, patches = [], []
    boxes, points = detect_face.detect_face(fixDim(image), minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    for b in boxes:
        x1, y1, x2, y2 = max(0, min(int(b[0]), image.shape[1])), max(0, min(int(b[1]), image.shape[0])), max(0, min(int(b[2]), image.shape[1])), max(0, min(int(b[3]), image.shape[0]))
        faces.append(image[y1:y2, x1:x2, :])
        patches.append(facenet.prewhiten(cv2.resize(image[y1:y2, x1:x2, :], (image_size, image_size), interpolation=cv2.INTER_AREA)))
    batch_size = len(patches)
    bc_indices, bc_probabilities = classifier.classify(patches, batch_size, model, sess, images_ph, embs, phase_train_ph, emb_size)
    labels = []
    for class_idx in bc_indices:
        labels.append(class_names[class_idx])
    return boxes, faces, labels, bc_probabilities




def bulk_main():

    pnet_fun, rnet_fun, onet_fun = initMTCNN()
    sess, images_ph, embs, phase_train_ph, emb_size, model, class_names = loadModel()

    images = []

    for file in os.listdir("/Users/Mohamad/Sensifai/FaceNet/data/aligned_images/Unknown/"):
        image = cv2.imread("/Users/Mohamad/Sensifai/FaceNet/data/aligned_images/Unknown/" + file)
        images.append(image[:, :, 0:3])

    nrof_images = 852
    nrof_batches = (nrof_images // batch_size) + 1
    for i in range(nrof_batches):
        if i == nrof_batches - 1:
            n = nrof_images
        else:
            n = i * batch_size + batch_size

        # detect faces in given image and return bounding boxes
        boxes = bulk_detect_faces_image(images[i * batch_size:n], pnet_fun, rnet_fun, onet_fun)


def bulk_detect_faces_image(images, pnet_fun, rnet_fun, onet_fun):
    start_time = time.time()
    boxes_points = detect_face.bulk_detect_face(images, 1/8, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    return boxes_points


def initMTCNN():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            return pnet, rnet, onet

def loadModel():
    sess, images_ph, embs, phase_train_ph, emb_size, model, class_names = classifier.initModel(model_dir, classifier_filename)
    return sess, images_ph, embs, phase_train_ph, emb_size, model, class_names


def fixDim(image):
    if image.ndim == 2:
        image = facenet.to_rgb(image)
    return image[:, :, 0:3]


if __name__ == '__main__':
    main()
    # bulk_main()


