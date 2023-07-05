import os
from tqdm import tqdm
import argparse
from os.path import join, isfile, isdir, basename, dirname, exists
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from shutil import copy2
import numpy as np
from time import time

from utils import load_txt, calc_sim, read_caps
from sentence_transformers import SentenceTransformer, util

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class VideoFrames(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.paths = sorted([join(root, p) for p in os.listdir(root) if isfile(join(root, p))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = Image.open(self.paths[item])
        # return self.transform(image).unsqueeze(0)
        image = self.transform(image)
        name = basename(self.paths[item])
        # save_image(image.squeeze(), join('/home/john/Desktop/test', name))

        return image, name


def remove_similar(img_dir, batch_size=4, nn=6):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Prepare the inputs
    dataset = VideoFrames(img_dir, _transform([224, 224]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Calculate features
    all_image_feats = torch.zeros(len(dataset), 512, dtype=torch.float16).to(device)
    all_image_names = []
    max_iter = len(dataset) // batch_size
    with torch.no_grad():
        for batch, (image_inputs, image_names) in tqdm(enumerate(data_loader)):
            image_features = model.encode_image(image_inputs.to(device))
            if batch <= max_iter:
                all_image_feats[batch * batch_size:(batch + 1) * batch_size, :] = image_features
            else:
                all_image_feats[batch * batch_size:, :] = image_features
            all_image_names.extend(image_names)

    all_image_feats /= all_image_feats.norm(dim=-1, keepdim=True)
    cosine_scores = util.cos_sim(all_image_feats, all_image_feats)
    inds_to_remove = []
    for i in range(len(cosine_scores)):
        for ind_, v in enumerate(cosine_scores[i]):
            if v >= 0.9 and ind_ > i:
                inds_to_remove.append(ind_)

    return list(set(inds_to_remove))
    # inds_to_remove = np.asarray(list(set(inds_to_remove)))
    # if len(inds_to_remove) > 0:
    #     images_after_removal = np.delete(np.asarray(all_image_names), inds_to_remove, None)
    #
    # with open(list_file, 'w') as fopen:
    #     for i in images_after_removal.tolist():
    #         fopen.write(i + '\n')
    #     fopen.close()



def get_args():
    parser = argparse.ArgumentParser('Extract images selected by hecate')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--img_dir', '-i', type=str,
                        help='path to all the images')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    remove_similar(img_dir='/raid/P15/4-data/mediacorp/The_Wish/DAP22804_CU/post_hecate',
                   batch_size=4,
                   )

