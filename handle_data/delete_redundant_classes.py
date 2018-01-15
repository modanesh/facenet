import shutil

base_dir = "/media/deepface/5a858105-5c78-47d2-b190-7fdf640e89b6/MS-Celeb-1M-Aligned-Faces/data/"
aligned_file = open("/Users/Mohamad/Sensifai/FaceNet/handle_align/aligned.txt")
data_file = open("/Users/Mohamad/Sensifai/FaceNet/handle_align/data.txt")

aligned_list = [line.rstrip() for line in aligned_file]
data_list = [line.rstrip() for line in data_file]

for folder_index in range(len(aligned_list)):
    if aligned_list[folder_index] in data_list:
        shutil.rmtree(base_dir + aligned_list[folder_index])
        print("deleting:", base_dir + aligned_list[folder_index])
