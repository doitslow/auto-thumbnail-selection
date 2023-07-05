import os
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
    def __init__(self, root, list_file, transform):
        self.transform = transform
        # self.paths = sorted([join(root, p) for p in os.listdir(root) if isfile(join(root, p))])
        self.paths = sorted([join(root, line) for line in open(list_file, "r").read().splitlines() if line][1:])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = Image.open(self.paths[item])
        # return self.transform(image).unsqueeze(0)
        image = self.transform(image)
        name = basename(self.paths[item])
        # save_image(image.squeeze(), join('/home/john/Desktop/test', name))

        return image, name


def keysentence2image(img_dir, image_list_file, out_dir, keysen, batch_size, topk=10):
    # keys = load_txt(join(dirname(img_dir), 'yake-keywords_in_sentence.txt'))
    # keysentence = [key.strip() for key in keys][0]
    keysentence = keysen.strip()

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Prepare the inputs
    dataset = VideoFrames(img_dir, image_list_file, _transform([224, 224]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    text_input = torch.cat([clip.tokenize(keysentence)]).to(device)

    # Calculate features
    all_image_feats = torch.zeros(len(dataset), 512, dtype=torch.float16).to(device)
    all_image_names = []
    max_iter = len(dataset) // batch_size
    with torch.no_grad():
        text_feature = model.encode_text(text_input)
        for batch, (image_inputs, image_names) in enumerate(data_loader):
            image_features = model.encode_image(image_inputs.to(device))
            if batch <= max_iter:
                all_image_feats[batch * batch_size:(batch + 1) * batch_size, :] = image_features
            else:
                all_image_feats[batch * batch_size:, :] = image_features
            all_image_names.extend(image_names)

    # Pick the top 5 most similar labels for the image
    all_image_feats /= all_image_feats.norm(dim=-1, keepdim=True)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    cosine_scores = torch.mm(text_feature, all_image_feats.transpose(0, 1))
    cosine_scores = cosine_scores.cpu().detach().numpy().squeeze()
    cosine_scores = (cosine_scores + 1) / 2
    # print(cosine_scores)
    print("The max/min similarity scores comparing captions to descriptions"
          " is: {:.4f}/{:.4f}".format(np.max(cosine_scores), np.min(cosine_scores)))
    topk_inds = np.argpartition(cosine_scores, -topk)[-topk:]

    if not exists(out_dir):
        os.mkdir(out_dir)
    reason_file = join(out_dir, 'm3-reasoning.txt')
    reasons = []
    with open(reason_file, 'w') as fopen:
        fopen.write("Keywords' sentence of description is: " + "\n")
        fopen.write(keysentence + '\n')
        fopen.write('\n')
        for i in topk_inds.tolist():
            reasons.append(all_image_names[i])
            copy2(join(img_dir, all_image_names[i]), out_dir)
            fopen.write(all_image_names[i] + '\t' + "{:.4f}".format(cosine_scores[i]) + '\n')
        fopen.close()

    return cosine_scores, reasons


def get_args():
    parser = argparse.ArgumentParser('Extract images selected by hecate')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--img_dir', '-i', type=str,
                        help='path to all the images')
    args = parser.parse_args()
    return args


