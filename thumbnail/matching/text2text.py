import os
import torch
import collections
import numpy as np
from shutil import copy2
from os.path import join, basename, dirname, exists

from sentence_transformers import SentenceTransformer, util
from utils import load_txt, calc_sim, read_caps


def caps_to_keysentence(image_dir, caption_file, out_dir, keysen, topk=10):
    caps, caps_ind = read_caps(caption_file)
    # keys = load_txt(join(video_folder, 'yake-keywords_in_sentence.txt'))
    keysen = [key.strip() for key in [keysen]]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings1 = model.encode(keysen, convert_to_tensor=True)
    embeddings2 = model.encode(caps, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    cosine_scores = cosine_scores.cpu().detach().numpy().squeeze()
    cosine_scores = (cosine_scores + 1) / 2
    print("The max/min similarity scores comparing captions to descriptions"
          " is: {:.4f}/{:.4f}".format(np.max(cosine_scores), np.min(cosine_scores)))
    topk_inds = np.argpartition(cosine_scores, -topk)[-topk:]

    # image_dir = join(video_folder, image_foldname)
    # out_dir = join(video_folder, 'm1-captions_to_keysentence-{}'.format(image_foldname))
    reason_file = join(out_dir, 'm1-reasoning.txt')
    if not exists(out_dir):
        os.mkdir(out_dir)
    reasons = []
    for ind in topk_inds:
        img = caps_ind[ind]
        copy2(join(image_dir, img), out_dir)
        reason = "{}\t Caption:{}\t Score: {:.4f}".format(img, caps[ind], cosine_scores[ind])
        reasons.append(reason)
    with open(reason_file, 'w') as fopen:
        fopen.write("Keywords' sentence of description is: " + "\n")
        fopen.write(keysen[0] + '\n')
        fopen.write('\n')
        for item in reasons:
            fopen.write(item + '\n')
        fopen.close()

    torch.cuda.empty_cache()

    return cosine_scores, reasons


def caps_to_keys(image_dir, caption_file, out_dir, keys, topk=5):   # not in use
    caps, caps_ind = read_caps(caption_file)
    # keys = load_txt(join(video_folder, 'yake-keywords.txt'))
    keys = [key.strip() for key in keys]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(keys, convert_to_tensor=True)
    embeddings2 = model.encode(caps, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    cosine_scores = cosine_scores.cpu().detach().numpy()

    all_inds = []
    all_scores = []
    topk_inds = np.zeros((cosine_scores.shape[0], topk), dtype=int)
    topk_scores = np.zeros((cosine_scores.shape[0], topk))
    for i in range(cosine_scores.shape[0]): # for each keyword
        scores = cosine_scores[i,:].squeeze()
        inds = np.argpartition(scores, -topk)[-topk:]
        all_inds.extend(list(inds))
        all_scores.extend(scores[inds])

        topk_inds[i, :] = inds
        topk_scores[i, :] = scores[inds]

    dups = [item for item, count in collections.Counter(list(topk_inds.flatten())).items() if count > 1]
    # image_dir = join(video_folder, image_foldname)
    # out_dir = join(video_folder, 'm2-captions_to_keywords-{}'.format(image_foldname))
    if not exists(out_dir):
        os.mkdir(out_dir)
    reason_file = join(out_dir, 'm2-reasoning.txt')

    reasons = []
    for dup in dups:
        locs = np.where(topk_inds == dup)
        c_keys = [keys[i] for i in list(locs[0])]
        scores = topk_scores[locs]
        img = caps_ind[topk_inds[locs][0]]
        copy2(os.path.join(image_dir, img), out_dir)
        scores = ["{:.4f}".format(score) for score in scores]
        reason = "{}\t Caption: {}\t Keywords: {}\t Scores: {}".format(
            img, caps[topk_inds[locs][0]], '; '.join(c_keys), '; '.join(scores))
        print(reason)
        reasons.append(reason)
        # for key, score in zip(c_keys, scores):
        #     print("{}\t {}\t {}\t Score: {:.4f}".format(img, caps[topk_inds[locs][0]], key, score))

    with open(reason_file, 'w') as fopen:
        fopen.write("Keywords of description are: " + "\n")
        fopen.write(";\t".join(keys) + '\n')
        fopen.write('\n')
        for item in reasons:
            fopen.write(item + '\n')
        fopen.close()

    torch.cuda.empty_cache()
