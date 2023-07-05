import os
import cv2
import argparse
from os.path import dirname, basename, join, exists
import shutil

from utils import count_jpeg


def vid_to_frame(video_path, save_dir, gap_in_seconds, display_img=False):
    if save_dir is None:
        save_dir = join(dirname(video_path), basename(video_path).split('.')[0])

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

        if exists(save_dir):
            num_images_to_take = int(frame_count // (origin_fps*gap_in_seconds)) + 1
            if num_images_to_take != count_jpeg(save_dir):
                print("Previous video to frame processing was not correct or completed, redoing!")
                shutil.rmtree(save_dir)
            else:
                # Release the video capture object
                vid_capture.release()
                cv2.destroyAllWindows()
                return

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
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
            if count % (origin_fps*gap_in_seconds) == 0:
                cv2.imwrite(os.path.join(save_dir, '{:06d}.jpeg'.format(count)), frame)
            # count += origin_fps
            # vid_capture.set(cv2.CAP_PROP_POS_MSEC, count)
            count += 1
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_video', '-p', type=str)
    parser.add_argument('--out_dir', '-o', type=str, default=None)
    parser.add_argument('--height', default=None, type=int)
    parser.add_argument('--width', default=None, type=int)
    parser.add_argument('--gap_in_seconds', default=4, type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    vid_to_frame(args.path_to_video, args.out_dir, args.gap_in_seconds)

