#!/usr/bin/env python

import os
import json

import torch

import liteflownet as lfn

from liteflownet.model import Network
from PIL import Image
from tqdm import tqdm

import logging

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

def get_all_video_ids(dataset_path, dataset_version, dataset_type):
    annotations_path = os.path.join(dataset_path, f'Diving48_{dataset_version}_{dataset_type}.json')
    with open(annotations_path, 'rb') as f:
        data = json.loads(f.read())

    return data

def write_flo_jpeg(flo, dirname):
    flo_rgb = flo.transpose(0, 2, 3, 1)
    flo_rgb = lfn.flowvid2rgb(flo_rgb)
    for i, frame in enumerate(flo_rgb):
        write_path = os.path.join(dirname, f'flow_{i+1:05d}.jpg')
        if os.path.exists(write_path):
            logger.warning(f'File {write_path} already exists, skipping')
            continue
        Image.fromarray(frame).save(write_path)

if __name__ == '__main__':
    DATASET_PATH = os.getenv('DATASET_PATH')
    DATASET_VERSION = os.getenv('DATASET_VERSION')
    DATASET_TYPE = os.getenv('DATASET_TYPE')
    OUTPUT_PATH = os.getenv('OUTPUT_PATH')

    assert DATASET_PATH is not None
    assert DATASET_VERSION is not None
    assert DATASET_TYPE is not None

    print(f'Starting script with the following parameters: \n\n\tDATASET_PATH={DATASET_PATH}\n\tDATASET_VERSION={DATASET_VERSION}\n\tDATASET_TYPE={DATASET_TYPE}')

    video_metas = get_all_video_ids(DATASET_PATH, DATASET_VERSION, DATASET_TYPE)

    logger.info(f'Found {len(video_metas)} videos')

    if not torch.cuda.is_available():
        logger.error(f'CUDA not available, exiting')
        exit(1)

    device = torch.device('cuda')
    model = Network().to(device)

    for video_meta in tqdm(video_metas):
        video_id = video_meta['vid_name']
        video_path = os.path.join(DATASET_PATH, 'rgb', f'{video_id}.mp4')
        assert os.path.exists(video_path), f'Video path does not exist: {video_path}'
        video = lfn.read_video_file(video_path)
        x = lfn.preprocess_video(video).to(device)
        flo = model(x[:-1], x[1:])
        write_flo_jpeg(flo, os.path.join(DATASET_PATH, 'flow', video_id))

