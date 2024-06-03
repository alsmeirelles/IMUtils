#!/usr/bin/env python3
# -*- coding: utf-8
# ALSMEIRELLES
import concurrent.futures
import os
import sys
import cv2
import argparse
import random
from tqdm import tqdm

from decord import VideoReader
from decord import cpu, gpu

# Searches for known video files in source directory
extensions = ['mp4', 'avi']


def run_annotated_frame_extraction(config: argparse.Namespace):
    """
    @param config: argparse.Namespace configuration object
    @return: number of extracted frames
    """
    extracted_frames = 0
    analyzed_videos = 0

    if os.path.isdir(config.vdata):
        input_files = list(filter(lambda x: x.split('.')[-1] in extensions, os.listdir(config.vdata)))
    else:
        return 0

    bar = tqdm(desc="Processing video files...\n", total=len(os.listdir(config.frames)), position=0)
    for f in input_files:
        fpath = os.path.join(config.frames, "{}.txt".format(f[:-4]))
        vpath = os.path.join(config.vdata, f)
        if not os.path.isfile(fpath):
            if config.verbose > 0:
                print(f"Skipping {f}, no annotation file found.")
            continue
        with open(fpath, "r") as fd:
            fns = fd.readlines()
        video_name = os.path.basename(vpath)[:-4]
        # vid = cv2.VideoCapture(vpath)
        vid = VideoReader(vpath, ctx=gpu(0))
        if config.verbose > 0:
            print(f"Video frame count: {len(vid)}")
            print(f"Video file: {video_name}")
        for line in fns:
            frame = line.strip().split(' ')
            if len(frame) < 2:
                continue
            frame = (int(frame[0]), int(frame[1]))
            read = list(range(max(0, frame[0] - config.nb), min(frame[0] + config.nb + 1, len(vid))))
            for findex in read:
                cv2.imwrite(os.path.join(config.out, '{}-{}_{}.jpg'.format(video_name, findex, frame[1])),
                            vid[findex].asnumpy())
        bar.update(1)


# noinspection PyTypeChecker
def run_frame_extraction(config: argparse.Namespace):
    """
    Searches for video files in data directory.
    For each video found, inspect every config.ns video frames every second
    @type config: argparse namespace object
    """
    extracted_frames = 0
    analyzed_videos = 0

    input_files = None
    # Check data source
    if os.path.isfile(config.fdata):
        input_files = [config.fdata]
    elif os.path.isdir(config.vdata):
        input_files = list(filter(lambda x: x.split('.')[-1] in extensions, os.listdir(config.vdata)))
    else:
        print(f"Video data not found in {config.vdata}{config.fdata}.")
        sys.exit(1)

    if config.cpu > 1 and len(input_files) > 1:
        from concurrent.futures import ThreadPoolExecutor

        bar = tqdm(desc="Processing video files...\n", total=len(input_files), position=0)
        videos = {}
        with ThreadPoolExecutor(max_workers=config.cpu) as executor:
            for f in input_files:
                ex = executor.submit(_run_extractor,
                                     os.path.join(config.vdata, f),
                                     config.out,
                                     config.ns,
                                     config.verbose,
                                     config.sufix)
                ex.add_done_callback(lambda x: bar.update(1))
                videos[ex] = f

        for task in concurrent.futures.as_completed(videos):
            extracted_frames += task.result()
            analyzed_videos += 1
    else:
        for f in input_files:
            if config.verbose:
                print(f"Extracting from {f}...")

            new_imgs = _run_extractor(os.path.join(config.vdata, f), config.out, config.ns, config.verbose,
                                      config.sufix)
            extracted_frames += new_imgs
            analyzed_videos += 1

    print("Extracted {} frames from {} videos".format(extracted_frames, analyzed_videos))


def _run_extractor(video_path: str, dst_dir: str, nfps: int, verbosity: int = 0, sufix: str = ''):
    """
    Runs extraction
    """
    random.seed()

    with open(video_path, "rb") as v:
        vid = VideoReader(v, ctx=cpu(0))

    video_name = os.path.basename(video_path)[:-4]
    vid_frames = len(vid)  # int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get_avg_fps())
    if verbosity >= 1:
        print(f"\nTotal frames in video: {vid_frames}; {fps} FPS")

    fcount = 0
    extracted = 0
    while fcount < vid_frames:
        pop = range(fcount, min(fcount + fps, vid_frames))
        fi = random.sample(pop, k=nfps) if len(pop) > nfps else pop

        for j in fi:
            cv2.imwrite(os.path.join(dst_dir, '{}-{}{}.jpg'.format(video_name, j, sufix)),
                        cv2.cvtColor(vid[j].asnumpy(), cv2.COLOR_BGR2RGB))
            extracted += 1

        fcount += fps

    # vid.release()
    return extracted


if __name__ == "__main__":

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract frames from video files.')

    parser.add_argument('-vdata', dest='vdata', type=str, default='',
                        help='Path to folder containing video files.', required=False)
    parser.add_argument('-fdata', dest='fdata', type=str, default='',
                        help='Path to a specific video file.', required=False)
    parser.add_argument('-frames', dest='frames', type=str, default='',
                        help='Path to dir that has .txt files with frame numbers to extract.', required=False)
    parser.add_argument('-out', dest='out', type=str, default='',
                        help='Save frames here.', required=True)
    parser.add_argument('-nb', dest='nb', type=int, default=1,
                        help='Combined with frame number annotation, extract nb frames before and after).')
    parser.add_argument('-ns', dest='ns', type=int, default=3,
                        help='Extract this many frames per second (Set to zero for all).')
    parser.add_argument('-cpu', dest='cpu', type=int, default=1,
                        help='Multiprocess extraction. Define the number of processes (Default=1).')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-sf', dest='sufix', type=str, default='',
                        help='Append sufix to output frames as _sufix.', required=False)

    config, unparsed = parser.parse_known_args()

    if not config.vdata and not config.fdata:
        print('You should define a path to a dir (vdata) or file (fdata)')
        sys.exit(1)

    if not os.path.isdir(config.out):
        os.mkdir(config.out)

    if config.frames:
        run_annotated_frame_extraction(config)
    else:
        run_frame_extraction(config)
