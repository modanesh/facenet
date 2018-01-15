data_file = open("/Users/Mohamad/Sensifai/FaceNet/handle_align/aaaa/data.txt")
aligned_data_file = open("/Users/Mohamad/Sensifai/FaceNet/handle_align/aaaa/aligned.txt")

data = []
aligned_data = []

for line in data_file.readlines():
    data.append(line[:-1])

for line in aligned_data_file.readlines():
    aligned_data.append(line[:-1])

not_common = []
not_common.extend(list(set(data) ^ set(aligned_data)))

print(len(not_common))
for i in range(len(not_common)):
    print(not_common[i])