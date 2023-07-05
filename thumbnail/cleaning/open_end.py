import os
import cv2
import sys
import math
from os.path import join, dirname, basename
from copy import deepcopy

# import mmocr
from cleaning.mmocr.mmocr.utils.ocr import MMOCR
from inaSpeechSegmenter import Segmenter


"""
    # For opening introduction, we handle two type of scenarios:
        1. long introduction with duration > 90s
        2. short introduction that is close to time 0 (>25s)
"""


class OpenEnd(object):
    def __init__(self, vid_path, open_duration=5, end_duration=3, batch_size=2, overlap_threshold=0.75):
        self.vid_path = vid_path
        self.open_duration = open_duration
        self.end_duration = end_duration
        self.batch_size = batch_size
        self.overlap_threshold = overlap_threshold

        self.cap = cv2.VideoCapture(vid_path)
        if self.cap.isOpened() == False:
            print("Error opening the video file")
        else:
            self.fps = self.cap.get(5)
            self.frame_count = int(self.cap.get(7))
            self.end_time = int(self.frame_count // self.fps)

        self.audio_segmenter = Segmenter(detect_gender=False)

    def run_ocr(self, din, ocr):
        fout = din + '-ocr.txt'
        if os.path.exists(fout):
            with open(fout, 'r') as fopen:
                lines = [line.strip().split('\t') for line in fopen.readlines()]
                outputs = [[line[0], [] if line[1] == 'null' else line[1].split(';')]
                           for line in lines]

            return outputs
        else:
            imgs = sorted([join(din, f) for f in os.listdir(din) if f.endswith('.jpeg')])
            outputs = []
            max_iter = len(imgs) // self.batch_size
            for i in range(max_iter):
                img_list = imgs[i * self.batch_size:(i + 1) * self.batch_size]
                outs = ocr.readtext(img_list, batch_mode=True,
                                    det_batch_size=self.batch_size,
                                    recog_batch_size=self.batch_size)
                for out in outs:
                    outputs.append([out['filename'], out['text']])

            print(len(outputs), len(imgs))
            if len(imgs) % self.batch_size != 0:
                for i in range(max_iter * self.batch_size, len(imgs)):
                    out = ocr.readtext(imgs[i])[0]
                    outputs.append([out['filename'], out['text']])

            assert(len(outputs) == len(imgs)), "OCR process not completed!"

            with open(fout, 'w') as fopen:
                for item in outputs:
                    fopen.write(item[0] + '\t' + (";".join(item[1]) if item[1] else "null"))
                    fopen.write('\n')

            return outputs

    # @staticmethod
    def count_consecutives(self, texts, names):
        consecutives = []
        start = 0
        end = 0
        for i, text in enumerate(texts):
            if text:
                if start == 0:
                    start = i
                else:
                    pass
            else:
                if start != 0:
                    end = i
                    consecutives.append([int(int(names[start])/self.fps), end - start])
                    start = 0
                else:
                    pass

        if start != 0:
            consecutives.append([int(int(names[start])/self.fps), len(texts) - start - 2])

        return consecutives

    def get_open_end(self, gap_in_seconds=1):
        end_of_open = self.fps * self.open_duration * 60
        start_of_end = self.frame_count - self.fps * self.end_duration * 60

        opening = join(dirname(self.vid_path), 'opening')
        ending = join(dirname(self.vid_path), 'ending')
        if os.path.exists(opening) and os.path.exists(ending):
            return
        else:
            os.mkdir(opening)
            os.mkdir(ending)

        count = 0
        out_dir = opening
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if count % (self.fps * gap_in_seconds) == 0:
                    cv2.imwrite(join(out_dir,
                                     '{:06d}.jpeg'.format(count)), frame)
                count += 1
            else:
                break
            if count == end_of_open:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_of_end)
                count = int(start_of_end) - 1
                out_dir = ending

        # Release the video capture object
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def find_longest(durations):
        longest_chunks = []
        longest_duration = max([item[1] for item in durations])
        for item in durations:
            if item[1] == longest_duration:
                longest_chunks.append(item)

        return longest_chunks

    @staticmethod
    def get_music_timeline(segs):
        timeline = []
        start_time = math.floor(segs[0][1])
        end_time = math.floor(segs[-1][2])
        for seg in segs:
            start = math.floor(seg[1])
            end = math.floor(seg[2])
            for i in range(start, end):
                # timeline.append(True) if seg[0] == 'music' else timeline.append(False)
                if seg[0] == 'music':
                    timeline.append('music')
                elif seg[0] == 'speech':
                    timeline.append('speech')
                else:
                    timeline.append('others')

        assert end_time - start_time == len(timeline), "Music timeline calculating not correct!"

        return {'start_end': [start_time, end_time], 'timeline': timeline}

    def subtitle_and_music(self, text_time, music_time):
        text_start = text_time[0]
        text_end = text_time[1]
        music_timeline_start = music_time['start_end'][0]
        music_timeline = music_time['timeline']
        offset_start = text_start - music_timeline_start
        offset_end = text_end - music_timeline_start

        # get overlap rate
        overlap_rate = sum([1 if i == 'music' else 0
                            for i in music_timeline[offset_start:offset_end]]) / (offset_end - offset_start)

        # get the related music duration
        overlapped_chunks = []
        music_start = 0
        music_counter = 0
        gap_counter = 0
        speech_counter = 0
        for i in range(offset_start, offset_end):
            if music_timeline[i] == 'music':
                if music_counter == 0:
                    music_start = i
                music_counter += 1
                gap_counter = 0
            else:
                if music_counter != 0:
                    gap_counter += 1
                    if music_timeline[i] == 'speech':
                        speech_counter += 1
                if gap_counter > 3 or speech_counter >= 3: # if music stops for 3 seconds, we consider the music ended
                    overlapped_chunks.append([music_start + music_timeline_start, music_counter])
                    music_start = 0
                    music_counter = 0
                    gap_counter = 0
                    speech_counter = 0

        if music_start != 0:
            overlapped_chunks.append([music_start + music_timeline_start, music_counter])

        # print("Overlapped durations are ", overlapped_chunks)

        return overlap_rate, overlapped_chunks

    def remove_open(self):
        open_subtitles = self.run_ocr(join(dirname(self.vid_path), "opening"),
                                      MMOCR(det='TextSnake', recog='SAR'))  # backbone for TextSnake is R50
        copy_of_subtitles = deepcopy(open_subtitles)

        for item in copy_of_subtitles:
            if len(item[1]) == 1 and len(item[1][0]) <= 3:
                item[1] = []
        # count supportive neighbours
        for i in range(3, len(copy_of_subtitles) - 3):
            if not copy_of_subtitles[i][1]:
                nbs = [copy_of_subtitles[i + offset][1] for offset in range(1, 4)]
                nbs.extend([copy_of_subtitles[i - offset][1] for offset in range(1, 4)])
                if sum([1 if nb else 0 for nb in nbs]) >= 5:
                    copy_of_subtitles[i][1] = ['filled_manually']

        strike_counts = self.count_consecutives(texts=[i[1] for i in copy_of_subtitles],
                                                names=[i[0] for i in copy_of_subtitles])
        valid_strikes = []  # only entertain strike that are longer than 10s
        for item in strike_counts:
            if item[-1] > 10:
                valid_strikes.append(item)
        print("Valid strikes are ", valid_strikes)

        if valid_strikes:
            open_segs = self.audio_segmenter(self.vid_path, start_sec=0, stop_sec=self.open_duration * 60)
            music_timeline = self.get_music_timeline(open_segs)
            print("Audio segmentation for the opening ", open_segs)

            for strike in valid_strikes:
                overlap_rate, overlap_durations = self.subtitle_and_music(
                    [strike[0], sum(strike)], music_timeline)
                if not overlap_durations:
                    continue

                for overlap_duration in overlap_durations:
                    # if overlap > self.overlap_threshold or longest_music_chunks[0][1] > 90:
                    if overlap_duration[1] > 90:    # scenario 1
                        # set the music start time as the time to start removing
                        removal_start = overlap_duration[0]
                        removal_end = sum(overlap_duration)

                        # examine 5s beyond the end of music or towards end of timeline
                        extension_end = removal_end + 5 if removal_end + 5 < music_timeline['start_end'][1] \
                            else music_timeline['start_end'][1]
                        speech_counter = sum([1 if i == 'speech' else 0 for i in
                                              music_timeline['timeline'][removal_end:extension_end]])
                        if speech_counter == 0:
                            removal_end = extension_end

                        print("By Scenario 1: Removing from {}={} to {}={}".format(
                            removal_start, removal_start * self.fps, removal_end, removal_end * self.fps))

                        return [int(removal_start * self.fps),  int(removal_end * self.fps)]
                    else:   # scenario 2: within 20s; no speech more than 3s, mostly music
                        chunk_start = overlap_duration[0]
                        chunk_end = chunk_start + overlap_duration[1]
                        if chunk_start < 20:
                            # check if there is speech within the selected duration
                            speech_counter = sum([1 if i == 'speech' else 0 for i in
                                                  music_timeline['timeline'][:chunk_start]])

                            # check if there is >3 consecutive frames without subtitles
                            subtitles_selected = open_subtitles[:chunk_end]
                            null_counter = 0
                            null_strikes = []
                            for sub in subtitles_selected:
                                if not sub[1]:
                                    null_counter += 1
                                else:
                                    if null_counter != 0:
                                        null_strikes.append(null_counter)
                                    null_counter = 0

                            if speech_counter == 0 and max(null_strikes) < 3:
                                removal_end = chunk_end
                                extension_end = removal_end + 5 if removal_end + 5 < music_timeline['start_end'][1] \
                                    else music_timeline['start_end'][1]
                                speech_counter = sum([1 if i == 'speech' else 0 for i in
                                                      music_timeline['timeline'][removal_end:extension_end]])
                                if speech_counter == 0:
                                    removal_end = extension_end

                                print("By Scenario 2: Removing from {}={} to {}={}".format(
                                    0, 0, removal_end, removal_end * self.fps))

                                return [0, int(removal_end * self.fps)]

        return [None, None]

    def remove_end(self):
        start_sec = int(self.end_time - self.end_duration * 60)
        end_subtitles = self.run_ocr(join(dirname(self.vid_path), "ending"),
                                     MMOCR(det='PANet_IC15', recog='ABINet')  # backbone for PANet_IC15 is R18
                                     )
        copy_of_subtitles = deepcopy(end_subtitles)

        # # remove false positive text detection:
        # # single words that have less than 3 characters usually are false positive
        # for item in copy_of_subtitles:
        #     if len(item[1]) == 1 and len(item[1][0]) <= 3:
        #         item[1] = []
        #
        # # smoothen detections
        # for i in range(3, len(copy_of_subtitles) - 3):
        #     if not copy_of_subtitles[i][1]:
        #         nbs = [copy_of_subtitles[i + offset][1] for offset in range(1, 4)]
        #         nbs.extend([copy_of_subtitles[i - offset][1] for offset in range(1, 4)])
        #         # count supportive neighbours
        #         if sum([1 if nb else 0 for nb in nbs]) >= 5 \
        #                 and sum([len(nb) == 1 for nb in nbs]) < 5:
        #             copy_of_subtitles[i][1] = ['filled_manually']

        # remove false positive text detection: harsh mode
        for item in copy_of_subtitles:
            if len(item[1]) == 1:
                item[1] = []

        # smoothen detections
        for i in range(3, len(copy_of_subtitles) - 3):
            if not copy_of_subtitles[i][1]:
                nbs = [copy_of_subtitles[i + offset][1] for offset in range(1, 4)]
                nbs.extend([copy_of_subtitles[i - offset][1] for offset in range(1, 4)])
                # count supportive neighbours
                if sum([1 if nb else 0 for nb in nbs]) >= 5:
                    copy_of_subtitles[i][1] = ['filled_manually']

        strike_counts = self.count_consecutives(
            texts=[i[1] for i in copy_of_subtitles],
            names=[i[0] for i in copy_of_subtitles]
        )
        print("Counting consecutive frames with subtitles\n", strike_counts)

        # only entertain strike that are longer than 10s
        valid_strikes = []
        for item in strike_counts:
            if item[-1] > 10:
                valid_strikes.append(item)
        print("Valid strikes are ", valid_strikes)

        # +++++++++++++++++++ Introduction of MUSIC +++++++++++++++++++++++++
        end_segs = self.audio_segmenter(self.vid_path, start_sec=start_sec)
        print("Audio segmentation is \n", end_segs)
        music_timeline = self.get_music_timeline(end_segs)

        # check every valid strikes backwardly: start with the last
        while valid_strikes:
            max_time = max([s[0] for s in valid_strikes])
            for s in valid_strikes:
                if s[0] == max_time:
                    strike = s

            overlap_rate, overlap_chunks = self.subtitle_and_music(
                [strike[0], sum(strike)], music_timeline)
            if not overlap_chunks:
                valid_strikes.remove(strike)
                continue

            # determine if there is speech afterwards
            start_of_music_in_strike = min([chunk[0] for chunk in overlap_chunks])
            speech_counter = sum(
                [i == 'speech' for i in
                 music_timeline['timeline'][(start_of_music_in_strike - start_sec):]]
            )
            # determine if ends within 10s
            ends_within_10 = int(strike[0]) / self.fps + strike[1] > self.end_time - 10
            if ends_within_10 or speech_counter < 3:
                # # last check if there is an abrupt increase in length of text
                # for i, sub in enumerate(copy_of_subtitles):
                #     if int(int(sub[0]) / self.fps) == strike[0]:
                #         strike_start_ind = i
                # subtitles_selected = copy_of_subtitles[strike_start_ind:]
                # abrupt_change = None
                # for i in range(10, len(subtitles_selected)//2):
                #     if sum([len(subtitles_selected[i + j][1]) for j in range(10)]) / \
                #             sum([len(subtitles_selected[i - j][1]) for j in range(10)]) >= 2:
                #         abrupt_change = i
                # if abrupt_change:
                #     removal_start = int(int(subtitles_selected[abrupt_change][0]) / self.fps)
                # else:
                #     removal_start = start_of_music_in_strike

                removal_start = start_of_music_in_strike
                return [int(removal_start * self.fps), self.frame_count]

            # discard if strike does not satisfy condition
            valid_strikes.remove(strike)

        return [None, None]

    def do_remove(self):
        self.get_open_end()
        open_removals = self.remove_open()
        end_removals = self.remove_end()
        with open(self.vid_path.replace('.mp4', '') + '-removals.txt', 'w') as fopen:
            fopen.write("\t".join([str(i) for i in open_removals]))
            fopen.write('\n')
            fopen.write("\t".join([str(i) for i in end_removals]))

        return open_removals, end_removals


if __name__ == '__main__':
    # din = "/raid/P15/4-data/mediacorp/CTRL/DAQ32043"
    # din = "/raid/P15/4-data/mediacorp/CLIF"
    # din = "/raid/P15/4-data/mediacorp/Local_Fine_Produce/DAU24646_CU"
    # din = "/raid/P15/4-data/mediacorp/The_Wish/DAP22809_CU"
    din = "/raid/P15/4-data/test"
    # din = '/raid/P15/4-data/mediacorp/CLIF/DAQ26300'

    for root, dirs, files in os.walk(din):
        for file in files:
            if file.endswith('.mp4'):
                print("WORKING on ", file)
                if not os.path.exists(os.path.join(root, file.replace('.mp4', '') + '-removals.txt')):
                    runner = OpenEnd(os.path.join(root, file), 5, 3)
                    runner.do_remove()

    # runner = OpenEnd('/raid/P15/4-data/mediacorp/The_Wish/DAP22804_CU/DAP22804_CU.mp4', 5, 3)
    # runner.do_remove()
