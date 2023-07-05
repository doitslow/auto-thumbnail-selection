import os
import cv2
import argparse
from os.path import join, dirname, basename, exists, isdir
from shutil import copy2, rmtree

from utils import count_jpeg


def read_image_list(in_file):
    img_ids = []
    with open(in_file, 'r') as fopen:
        lines = fopen.readlines()
        for line in lines:
            words = line.split('\t')
            if len(words) < 3:
                total_no = int(words[0])
            else:
                img_ids.append(words[1])

    return total_no, img_ids


def get_hecate_images(video_path, aes_path, save_dir=None, display_img=False):
    # find image list file
    img_list = video_path + '.txt'
    if not exists(img_list):
        if exists(join(aes_path, basename(img_list))):
            copy2(join(aes_path, basename(img_list)), dirname(video_path))
        else:
            print("Hecate images list file not found, skipping!")
            return False
    num_imgs, img_ids = read_image_list(img_list)

    if save_dir is None:
        save_dir = join(dirname(video_path), 'post_hecate')
    if exists(save_dir):
        if count_jpeg(save_dir) == num_imgs:
            return True
        else:
            print("Number of images in post_hecate not correct, re-doing hecate frame extraction!")
            rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(video_path)

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        origin_fps = vid_capture.get(5)
        print('Frames per second : ', origin_fps, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    count = 0
    while (vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame

        ret, frame = vid_capture.read()
        if ret == True:
            if display_img:
                cv2.imshow('Frame', frame)
                # 20 is in milliseconds, try to increase the value, say 50 and observe
                key = cv2.waitKey(20)
                if key == ord('q'):
                    break

            if str(count) in img_ids:
                cv2.imwrite(os.path.join(save_dir, '{:06d}.jpeg'.format(count)), frame)
            count += 1
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()

    return True


def run_mediacorp(din):
    for root, dirs, files in os.walk(din):
        for file in files:
            if file.endswith('.mp4'):
                print("Found a folder with video file: {}".format(root))
                # First check out if there is a correpsonding list file
                img_list = join(root, file + '.txt')
                if not exists(img_list):
                    print("File that list out post hecate images is missing!")
                else:
                    hecate_folder = join(root, 'post_hecate')
                    if exists(hecate_folder) and len(os.listdir(hecate_folder)) > 0:
                        pass
                    else:
                        get_hecate_images(join(root, file), img_list)


def get_args():
    parser = argparse.ArgumentParser('Extract images selected by hecate')
    # parser.add_argument('--img_list', '-i', type=str, default=None,
    #                     help='path to the file that list images selected by hecate')
    parser.add_argument('--video_path', '-v', type=str, default=None,
                        help='path to the video')
    parser.add_argument('--aes_path', type=str, default="/raid/P15/4-data/aes")
    parser.add_argument('--din', type=str, default=None,)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.din is not None:
        run_mediacorp(args.din)
    else:
        if args.img_list is None:
            args.img_list = args.video_path + '.txt'
        get_hecate_images(args.video_path, args.aes_path)
