import os
import random
import shutil


msceleb_dir = "/home/deepface/users/danesh/FaceNet/validation_test/msceleb_validation/"
pairs_dir = "/home/deepface/users/danesh/FaceNet/validation_test/msceleb_pairs.txt"
ordered_path = "/home/deepface/users/danesh/FaceNet/validation_test/ordered_msceleb_pairs.txt"


classes = []
images = []

def rename_images():
    for folder in os.listdir(msceleb_dir):
        print(folder)
        classes.append(folder)
        count = 0
        for file in os.listdir(msceleb_dir + folder):
            images.append(file)
            print(file)
            count += 1
            os.rename(msceleb_dir + folder + "/" + file, msceleb_dir + folder + "/" + folder + "_" + str(count).zfill(4) + ".png")



def write_pair_file():
    pair_file = open(pairs_dir, "w")
    matched_pairs = []
    _, dirs, _ = os.walk(msceleb_dir).__next__()
    for i in range(len(dirs)):
        _, _, files = os.walk(msceleb_dir + dirs[i]).__next__()
        if len(files) > 1:
            for j in range(1, len(files)):
                if not j + 1 > len(files):
                    matched_pairs.append((dirs[i], str(j), ))
                    pair_file.write(dirs[i] + "\t" + str(j) + "\t" + str(j+1) + "\n")

    mismatched_pairs = []
    for i in range(len(matched_pairs)):
        _, dirs, _ = os.walk(msceleb_dir).__next__()
        class_rand1, class_rand2 = random.sample(range(0, len(dirs)), 2)
        _, _, files1 = os.walk(msceleb_dir + dirs[class_rand1]).__next__()
        _, _, files2 = os.walk(msceleb_dir + dirs[class_rand2]).__next__()
        file_rand1 = random.randint(1, len(files1))
        file_rand2 = random.randint(1, len(files2))
        mismatched_pairs.append((dirs[class_rand1], file_rand1, dirs[class_rand2], file_rand2))
        pair_file.write(dirs[class_rand1] + "\t" + str(file_rand1) + "\t" + dirs[class_rand2] + "\t" + str(file_rand2) + "\n")

    # return matched_pairs, mismatched_pairs

    print(len(matched_pairs))


def order_pair_file():
    lines = open(pairs_dir).readlines()

    new_pair_file = open(ordered_path, "w")

    for i in range(1000):
        for i in range(149):
            new_pair_file.write(lines[i])
            print(lines[i])
        for i in range(149):
            new_pair_file.write(lines[149000 + i])
            print(lines[149000 + i])








if __name__ == '__main__':
    # rename_images()
    write_pair_file()
    order_pair_file()