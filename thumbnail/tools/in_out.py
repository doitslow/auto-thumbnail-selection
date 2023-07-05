import os
import argparse
from os.path import dirname, basename, join, exists, isdir
from shutil import copy2, copytree, move

"""
The directory structure shall follow:
MediaCorp
    - Series Name: example1 <The Wish>
        - Episode 1
        - Episode 2
        ...
    - Series Name: example2 <CLIF>
        - Episode 1
        - Episode 2
        ...
    ...
"""


def create_folder_under_series(din):  # deals with some video that has extension of MP4
    for root, dirs, files in os.walk(din):
        for file in files:
            if file.lower().endswith('.mp4'):
                file_splits = file.split('.')
                file_splits[-1] = 'mp4'
                fname = '.'.join(file_splits)
                if file != fname:
                    move(join(din, file), join(din, fname))
                folder = fname.replace('.mp4', '')
                if not exists(join(din, folder)):
                    os.mkdir(join(din, folder))
                if not exists(join(din, folder, fname)):
                    move(join(din, fname), join(din, folder, fname))


def create_folder_for_each_video(din):
    series = [join(din, d) for d in os.listdir(din) if isdir(join(din, d))]
    print("The series are", series)
    for s in series:
        create_folder_under_series(s)


def take_results(din, dout):
    if not exists(dout):
        os.mkdir(dout)
    for root, dirs, files in os.walk(din):
        for file in files:
            if file.endswith('.mp4'):
                series = basename(dirname(root))
                episode = basename(root)
                out_folder = join(dout, series, episode)
                if not exists(join(dout, series)):
                    os.mkdir(join(dout, series))
                if not exists(out_folder):
                    os.mkdir(out_folder)

                for file in files:
                    if not file.endswith('.mp4'):
                        copy2(join(root, file), out_folder)
                for dir_ in dirs:
                    if dir_.startswith('m1') or dir_.startswith('m2') or dir_.startswith('m3'):
                        copytree(join(root, dir_), join(out_folder, dir_))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Organize videos into separate folders')
    parser.add_argument('--din', type=str, default='/raid/P15/4-data/mediacorp')
    parser.add_argument('--in_or_out', '-io', type=str, choices=['in', 'out'])
    parser.add_argument('--dout', type=str, default=None)
    args = parser.parse_args()
    if args.in_or_out == 'in':
        create_folder_for_each_video(args.din)
    else:
        take_results(args.din, join(args.dout, basename(args.din) + "_results"))

