import cv2
from PIL import Image
import os



def extract_frames(video_path, frames_path):
  vidcap = cv2.VideoCapture(video_path)
  success, image = vidcap.read()
  count = 0
  success = True

  while success:
    cv2.imwrite(frames_path + "frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

def rotate_frames(frames_path, rotated_path):
  for filename in os.listdir(frames_path):

    img = Image.open(frames_path + filename)

    img2 = img.rotate(+90, expand=True)
    img2.save(rotated_path + filename)


if __name__ == '__main__':
  frm_path = "/home/danesh/sens/sensid06/"
  vid_path = "/home/danesh/sens/sensid06.mp4"
  extract_frames(vid_path, frm_path)
