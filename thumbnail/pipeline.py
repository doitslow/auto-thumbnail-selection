import os
import sys
print(sys.path)
import argparse
from os.path import join, basename, exists
import numpy as np
from shutil import copy2

from utils import create_description_file, count_jpeg, read_description_file, \
    check_list_file, write_list_file


class SemanticMatch(object):
    def __init__(self, args):
        self.args = args
        self._work_dir = None

    @property
    def work_dir(self):
        return self._work_dir

    @work_dir.setter
    def work_dir(self, din):
        self._work_dir = din

    def get_keywords(self, video_path): # STEP1: extract keywords
        if not exists(join(self.work_dir, 'description.txt')):
            if not create_description_file(video_path, self.args.meta_path):
                print("Semantic matching cannot be done as description file cannot be found nor created !!!")
                return
        description, keywords, keywords_in_sentence = read_description_file(
            join(self.work_dir, 'description.txt'))
        if keywords is None or keywords_in_sentence is None:
            from keywords.yake_keywords import yake1file
            keywords, keysen = yake1file(join(self.work_dir, 'description.txt'))
        else:
            keywords = keywords.split('\t')[-1].split(';')
            keysen = keywords_in_sentence.split('\t')[-1]
        print("STEP1: Keywords generation done.")

        return keywords, keysen

    def vid_to_frame(self, video_path):
        if self.args.post_hecate:
            img_dir = join(self.work_dir, "post_hecate")
            from tools.hecate import get_hecate_images
            if not get_hecate_images(video_path, self.args.aes_path):
                print("Semantic matching cannot be done as Hecate images cannot be fetched !!!")
                return
        else:
            img_dir = join(self.work_dir, "images_every_{}s".format(self.args.gap_in_seconds))
            from tools.vid2img import vid_to_frame
            vid_to_frame(video_path, img_dir, self.args.gap_in_seconds)

        return img_dir

    def cleaning(self, video_path, img_dir):
        # Clean away unwanted images and create a list of remaining
        if args.cleaning is None:
            list_file = img_dir + '.txt'
            if not check_list_file(list_file):
                write_list_file(list_file, sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')]))
        else:
            list_file = img_dir + '-post_{}_cleaning.txt'.format(args.cleaning)
            inds_to_remove = []
            if exists(list_file) and check_list_file(list_file):
                pass
            else:
                # remove similar images
                from cleaning.similar_images import remove_similar
                inds_to_remove.extend(remove_similar(img_dir, batch_size=self.args.clean_batch))  # returns a list
                # remove opening/ending
                if args.cleaning == 'c2':
                    from cleaning.open_end import OpenEnd
                    runner = OpenEnd(video_path, batch_size=self.args.clean_batch)
                    open_removals, end_removals = runner.do_remove()
                    all_hecate_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])
                    for ind_, img in enumerate(all_hecate_images):
                        frame_id = int(img.replace('.jpeg', ''))
                        if open_removals[0] and open_removals[1]:
                            if open_removals[0] <= frame_id <= open_removals[1]:
                                print("Removing opening frame: ", frame_id)
                                inds_to_remove.append(ind_)
                        if end_removals[0] and end_removals[1]:
                            if end_removals[0] <= frame_id <= end_removals[1]:
                                print("Removing ending frame: ", frame_id)
                                inds_to_remove.append(ind_)

                uniq_inds_to_remove = list(set(inds_to_remove))
                all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])
                if len(uniq_inds_to_remove) == 0:
                    print("NOT removing any images in preprocess")
                    images_post_clean = all_images
                else:
                    images_post_clean = list(
                        np.delete(np.asarray(all_images), np.asarray(uniq_inds_to_remove), None))
                write_list_file(list_file, images_post_clean)
        print("STEP2: Finalising candidate frames done.")

        return list_file

    def match1(self, img_dir, list_file, keysen):
        caption_file = '{}-captions.txt'.format(list_file.replace('.txt', ''))
        if not exists(caption_file) or \
                int(open(list_file, "r").readline().strip()) != len([line for line in open(caption_file, "r").read().splitlines() if line]):
            # from caption.caption_batch import caption_frames
            from ofa.caption_batch import caption_frames
            caption_frames(img_dir, list_file, batch_size=self.args.caption_batch, use_fp16=True)

        out_dir = join(self.work_dir, 'm1-captions_to_keysentence-{}'
                       .format(basename(list_file.replace('.txt', ''))))
        from matching.text2text import caps_to_keysentence
        scores, selection_with_reason = caps_to_keysentence(img_dir, caption_file, out_dir, keysen)

        return scores, selection_with_reason

    def match2(self, img_dir, list_file, keysen):
        out_dir = join(self.work_dir, 'm2-image_to_keysentence-{}'
                       .format(basename(list_file.replace('.txt', ''))))
        from matching.text2image import keysentence2image
        scores, selection_with_reason = keysentence2image(img_dir, list_file, out_dir, keysen, self.args.match_batch)

        return scores, selection_with_reason

    def run_single_video(self, video_folder):
        self.work_dir = video_folder
        print("Working on {} with {}-{}".format(
            self.work_dir, 'hecate' if self.args.post_hecate else 'gap', self.args.method))
        video_files = [d for d in os.listdir(self.work_dir) if d.endswith('.mp4')]
        assert len(video_files) > 0, "No video files found in {}".format(self.work_dir)
        assert len(video_files) == 1, "More than 1 video files found in {}".format(self.work_dir)
        video = join(self.work_dir, video_files[0])

        # ------------------------- Keywords --------------------------------------------
        _, keysentence = self.get_keywords(video)

        # ---------------- Clean away unwanted candidates generated by HECATE -----------
        img_dir = self.vid_to_frame(video)
        img_list_file = self.cleaning(video, img_dir)

        # -------------------------  DO MATCHING ----------------------------------------
        if self.args.method == 'm1':
            scores, selections = self.match1(img_dir, img_list_file, keysentence)
        elif self.args.method == 'm2':
            scores, selections = self.match2(img_dir, img_list_file, keysentence)
        elif self.args.method == 'en':
            topk = 10
            scores1, _ = self.match1(img_dir, img_list_file, keysentence)
            scores2, _ = self.match2(img_dir, img_list_file, keysentence)
            scores = 0.4 * scores1 + 0.6 * scores2
            topk_inds = np.argpartition(scores, -topk)[-topk:]
            imgs = sorted([line for line in open(img_list_file, "r").read().splitlines() if line][1:])

            out_dir = join(self.work_dir, 'ensemble-{}'
                           .format(basename(img_list_file.replace('.txt', ''))))
            if not exists(out_dir):
                os.mkdir(out_dir)
            reason_file = join(out_dir, 'ensemble-reasoning.txt')
            with open(reason_file, 'w') as fopen:
                fopen.write("Keywords' sentence of description is: " + "\n")
                fopen.write(keysentence + '\n')
                fopen.write('\n')
                for i in topk_inds.tolist():
                    copy2(join(img_dir, imgs[i]), out_dir)
                    fopen.write(imgs[i] + '\t' + "{:.4f}".format(scores[i]) + '\n')
                fopen.close()

            print(scores)

        print("STEP3: Matching done.")
        return scores

    def run(self, any_path):    # process all the videos in the given directory
        # parse directory
        video_folders = []
        for root, dirs, files in os.walk(any_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_folders.append(root)
        print("In this run, we are going to process {} videos".format(len(video_folders)))

        for folder in video_folders:
            self.run_single_video(folder)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_batch', type=int, default=32,
                        help='batch size')
    parser.add_argument('--caption_batch', type=int, default=2,
                        help="batch size for caption caption module")
    parser.add_argument('--clean_batch', type=int, default=4,
                        help="batch size for caption caption module")
    parser.add_argument('--din', "-d", type=str, default=None)
    parser.add_argument('--post_hecate', action='store_true',
                        help="use image selected by hecate")
    parser.add_argument('--gap_in_seconds', default=4, type=int)
    parser.add_argument('--method', '-m', default='en', choices=['m1', 'm2', 'en'],
                        help="m1: compare caption of each frame to one sentence which "
                             "is composed by concatenating keywords in sequential"
                             "m2: compare image to keyword sentence"
                             "en: ensemble of m1 and m2 by averaging scores"
                        )
    parser.add_argument('--aes_path', type=str, default="/raid/P15/4-data/aes")
    parser.add_argument('--meta_path', type=str, default='/raid/P15/4-data/mediacorp/Metadata.xlsx')
    parser.add_argument('--cleaning', type=str, choices=[None, 'c1', 'c2'], default=None,
                        help='c1: remove similar images'
                             'c2: remove similar images + remove opening/ending parts')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    matcher = SemanticMatch(args)
    matcher.run(args.din)
