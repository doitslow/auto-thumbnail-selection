import os
from os.path import join, basename, dirname, abspath
import sys
print(sys.path)
# sys.path.append(dirname(dirname(abspath(__file__))))  # don't do this
import torch
import string
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile

from fairseq import utils, tasks
from fairseq import checkpoint_utils
from fairseq.logging import meters, metrics, progress_bar
from fairseq.data import iterators, FairseqDataset
# from caption.utils.eval_utils import eval_step
# from caption.tasks.mm_tasks.caption import CaptionTask
# from caption.models.caption import OFAModel
# from caption.data import data_utils
# from caption.data.ofa_dataset import OFADataset


from .utils.eval_utils import eval_step
from .tasks.mm_tasks.caption import CaptionTask
from .models.ofa import OFAModel
from .data import data_utils
from .data.ofa_dataset import OFADataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
        },
    }

    return batch


class CaptionDataset(OFADataset):
    def __init__(
        self,
        split,  # not in use
        dataset,
        image_list_file,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)

        # get all image paths
        # print(image_list_file)
        # print([line for line in open(image_list_file, "r").read().splitlines() if line].pop(0))
        self.images = sorted([join(dataset, line) for line in open(image_list_file, "r").read().splitlines() if line][1:])
        # self.images = []
        # for root, dirs, files in os.walk(self.dataset):
        #     for file in files:
        #         if file.endswith('.jpeg'):
        #             self.images.append(join(root, file))
        # self.images = sorted(self.images)
        print("Size of {}: {}".format(dataset, len(self.images)))

        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst
        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # overwrite length of dataset as we are directly looping through directory
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        uniq_id = basename(self.images[index])
        image = Image.open(self.images[index])

        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        src_item = self.encode_text(" what does the image describe?")
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)


def get_dataloader(img_folder, image_list_file, cfg, task, batch_size, num_workers, buffer_size):
    dataset = CaptionDataset(
        None,
        img_folder,
        image_list_file,
        task.bpe,
        task.src_dict,
        task.tgt_dict,
        patch_image_size=cfg.task.patch_image_size,
    )
    # create mini-batches with given size constraints
    max_sentences = batch_size
    num_shards = 1
    import math
    batch_sampler = [
        [j for j in range(i, min(i + max_sentences, len(dataset)))]
        for i in range(0, len(dataset), max_sentences)
    ]
    assert isinstance(dataset, FairseqDataset)

    # initialize the dataset with the correct starting epoch
    dataset.set_epoch(1)
    total_row_count = len(dataset.images)
    num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
    if len(batch_sampler) < num_batches:
        batch_sampler.append([])
    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=7,
        num_shards=1,
        shard_id=0,
        num_workers=num_workers,
        epoch=1,
        buffer_size=buffer_size
    )

    return epoch_iter


def caption_frames(din, image_list_file, num_workers=4, batch_size=2, buffer_size=20, use_fp16=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Register caption task
    tasks.register_task('caption', CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    if not use_cuda:
        use_fp16 = False

    ofa_dir = dirname(abspath(__file__))
    # Load pretrained ckpt & config
    overrides={"bpe_dir":"{}/utils/BPE".format(ofa_dir),
               "eval_cider":False,
               "beam":5,
               "max_len_b":16,
               "no_repeat_ngram_size":3,
               "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('{}/checkpoints/caption_base_best.pt'.format(ofa_dir)),
            arg_overrides=overrides
        )

    # Move models to GPU
    # print(cfg.distributed_training.pipeline_model_parallel)
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    output = []
    epoch_itr = get_dataloader(din, image_list_file, cfg, task, batch_size, num_workers, buffer_size)
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=False,
    )
    itr = iterators.GroupedIterator(itr, 20)
    data_loader = progress_bar.progress_bar(itr)
    # Run eval step for caption
    with torch.no_grad():
        for i, samples in enumerate(data_loader):
            for sample in samples:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
                result, scores = eval_step(task, generator, models, sample)
                output.extend(result)
                print(result)
    torch.cuda.empty_cache()

    outfile = image_list_file.replace('.txt', '') + '-captions.txt'
    with open(outfile, 'w') as fopen:
        for line in output:
            fopen.write('\t'.join([line['image_id'], line['caption']]) + '\n')
        fopen.close()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":
    # test
    caption_frames('/raid/P15/4-data/mediacorp/Local_Fine_Produce/DAU24651_CU/images_every_4s',
                   '/raid/P15/4-data/4x4/CLIF/DAQ25422/images_every_4s-post_c1_cleaning.txt',
                   batch_size=1,
                   buffer_size=10,
                   use_fp16=True,
                   )
