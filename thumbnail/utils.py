import os
import pandas
import re
import numpy as np
from os.path import dirname, basename, join, exists


def load_txt(file_in):
    with open(file_in, 'r') as fopen:
        lines = fopen.readlines()
        fopen.close()

    return lines


def calc_sim(vA, vB):
    return np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))


def read_caps(file):
    with open(file, 'r') as fopen:
        caps_data = fopen.readlines()
    caps_ind, caps = [], []
    for item in caps_data:
        i, c = item.strip().split('\t')
        caps_ind.append(i)
        caps.append(c)

    return caps, caps_ind


def count_jpeg(din):
    jpegs = [f for f in os.listdir(din) if f.endswith('.jpeg')]

    return len(jpegs)


def create_description_file(vid, meta_file):
    folder = dirname(vid)
    vid_name = basename(vid)
    df = pandas.read_excel(meta_file)
    video_row = df.loc[df['Video Filename'] == vid_name.upper()]
    if video_row.empty:
        print("Video not found in meta file!!!")
        return False
    else:
        text = video_row['Summary'].values[0]
        # text = df.loc[df['Video Filename'] == vid_name.upper()]['Summary'].values[0]
        start_of_chinese = re.search(r'[\u4e00-\u9fff]+', text).start()
        english_description = text[:start_of_chinese]
        print("After removing chinese part: \n", english_description)
        with open(join(folder, 'description.txt'), 'w') as fopen:
            fopen.write(english_description)
        return True


def read_description_file(fpath):
    with open(fpath, 'r') as fopen:
        lines = fopen.readlines()
        if len(lines) == 0:
            print("Description file is empty!")
            return None, None, None
        elif len(lines) == 1:
            print("Keywords have not been generated!")
            return lines[0], None, None
        elif len(lines) == 3:
            print("Description and keywords are there!")
            return lines[0], '\t'.join(lines[1].split('\t')[1:]), lines[2].split('\t')[-1]
        else:
            print("Description file is kind of messy, please open and check!")
            return None, None, None


def check_list_file(fin):
    # assumes that the first line indicates the total number of items
    items = [line for line in open(fin, "r").read().splitlines() if line]
    print("There shall be {} items, there are {} items in {}".format(items[0], len(items) - 1, fin))

    return int(items[0]) == len(items) - 1


def write_list_file(fout, items):
    with open(fout, 'w') as fopen:
        fopen.write(str(len(items)) + '\n')
        for item in items:
            fopen.write(item + '\n')
        fopen.close()


def read_image_list(fin):
    items = [line for line in open(fin, "r").read().splitlines() if line][1:]

    return items

