import os

directory = "/Users/Mohamad/Sensifai/FaceNet/data/aligned_images/"
threshold = 100

for folder in os.listdir(directory):
    count = 0
    for file in os.listdir(directory + folder):
        if file.endswith("jpg") or file.endswith("png"):
            count += 1

    if count <= threshold:
        print(folder)
        print(count)
