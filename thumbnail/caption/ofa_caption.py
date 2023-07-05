import os
from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from ofa.utils.eval_utils import eval_step
from ofa.tasks.mm_tasks.caption import CaptionTask
from ofa.models.ofa import OFAModel
from PIL import Image

# Register caption task
tasks.register_task('caption', CaptionTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

ofa_dir = "/raid/P15/2-code/auto_thumbnail_selection/caption"
# Load pretrained ckpt & config
overrides={"bpe_dir":"{}/utils/BPE".format(ofa_dir),
           "eval_cider":False,
           "beam":5,
           "max_len_b":16,
           "no_repeat_ngram_size":3,
           "seed":7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        # utils.split_paths('checkpoints/caption_base_best.pt'),
        utils.split_paths('{}/checkpoints/caption_base_best.pt'.format(ofa_dir)),
        arg_overrides=overrides
    )

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def caption_frames(img_folder):
    output = []
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.endswith('.jpeg'):
                image = Image.open(os.path.join(root, file))

                # Construct input sample & preprocess for GPU if cuda available
                sample = construct_sample(image)
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

                # Run eval step for caption
                with torch.no_grad():
                    result, scores = eval_step(task, generator, models, sample)

                print('Caption: {}'.format(result[0]['caption']))
                output.append([file, result[0]['caption']])

    outfile = join(dirname(img_folder), '{}_captions.txt'.format(basename(img_folder)))
    with open(outfile, 'w') as fopen:
        for line in output:
            fopen.write('\t'.join(line) + '\n')
        fopen.close()