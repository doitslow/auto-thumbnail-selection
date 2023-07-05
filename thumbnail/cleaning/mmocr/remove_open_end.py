# Check Pytorch installation
import sys
print(sys.path)
import torch

print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(mmcv.__version__)
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check mmocr installation
from cleaning import mmocr

print(mmocr.__version__)

# %cd /mmocr/
# !ls

from cleaning.mmocr import MMOCR
# ocr = MMOCR(det=None, recog='SAR')
# Load models into memory
ocr = MMOCR(det='TextSnake', recog='SAR')
# ocr = MMOCR()
# ocr.readtext('/home/john/Desktop/003200.jpeg', print_result=True, output='outputs/demo_text_recog_pred.jpg')
import os
from os.path import join
#
# din = "/raid/P15/4-data/4x4/Local_Fine_Produce/DAU24645_CU/post_hecate"
din = "/home/john/Desktop/ending"


def run_ocr(din):
    imgs = sorted([join(din, f) for f in os.listdir(din) if f.endswith('.jpeg')])
    outputs = []
    for img in imgs:
        out = ocr.readtext(img, print_result=True)[0]
        outputs.append([out['filename'], out['text']])

    fout = din + '-ocr.txt'
    with open(fout, 'w') as fopen:
        for item in outputs:
            fopen.write(item[0] + '\t' + (";".join(item[1]) if item[1] else "null"))
            fopen.write('\n')

    count_consecutives(texts=[i[1] for i in outputs], names=[i[0] for i in outputs])


def count_consecutives(texts, names):
    consecutives = {}
    start = 0
    end = 0
    count = 0
    for i, text in enumerate(texts):
        print(text)
        # if text != 'null':
        if text != 'none':
            if start == 0:
                start = i
            else:
                pass
        else:
            if start != 0:
                end = i
                consecutives[names[start]] = end - start
                start = 0
            else:
                pass
    print(consecutives)

    return consecutives


names = []
texts = []
with open('/home/john/Desktop/opening-ocr.txt', 'r') as fopen:
    lines = fopen.readlines()
    for line in lines:
        line = line.strip()
        names.append(line.split('\t')[0])
        texts.append(line.split('\t')[1])

    count_consecutives(texts, names)


# run_ocr(din)