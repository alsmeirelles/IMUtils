#!/usr/bin/env python3
# -*- coding: utf-8
# ALSMEIRELLES
import concurrent.futures
import os
import sys
import cv2
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from decord import VideoReader
from decord import cpu, gpu

# Locals
from .CropRegion import crop_color_region
from .ImageOps import image_rotate

# Searches for known video files in source directory
extensions = ['mp4', 'avi']

def preprocess_frame(
    frame_bgr: np.ndarray,
    *,
    force_landscape: bool = True,
    crop_region: bool = False,
    region_bgr: Optional[Tuple[int, int, int]] = None,
    region_tol_h: int = 16,
    region_tol_s: int = 80,
    region_tol_v: int = 80,
    region_margin_px: int = 24,
    region_min_area_frac: float = 0.05,
    region_close_x_frac: float = 1/18,
    region_close_y_frac: float = 1/120,
    region_row_thresh: Optional[float] = None,
    region_row_min_run_frac: float = 0.35,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "frame",
) -> np.ndarray:
    """
    Fast, in-memory preprocessing before saving a frame.

    Steps:
      1) Optional: rotate to landscape (width > height).
      2) Optional: crop the largest region that matches the provided BACKGROUND color
         (e.g., the blue conveyor) + small margin. Pure crop (no resize), so aspect is preserved.

    Args:
        frame_bgr: Raw frame in BGR.
        force_landscape: If True, rotates 90° when H > W.
        crop_background_bgr: If given, perform color-based crop using this (B,G,R).
        crop_tol_*: HSV tolerances for robust color selection.
        crop_margin_px: Uniform expansion around the detected region.
        crop_min_area_frac: Reject tiny blobs.

    Returns:
        The processed BGR frame.
    """
    out = frame_bgr

    if force_landscape and out.shape[0] > out.shape[1]:
        out, _ = image_rotate(out, orientation="h",rnumpy=True, conditional=True)

    if crop_region:
        out, _ = crop_color_region(
            out,
            background_bgr=region_bgr,
            tol_h=region_tol_h,
            tol_s=region_tol_s,
            tol_v=region_tol_v,
            min_area_frac=region_min_area_frac,
            margin_px=region_margin_px,
            morph_close_x_frac=region_close_x_frac,
            morph_close_y_frac=region_close_y_frac,
            row_ratio_thresh=region_row_thresh,
            row_min_run_frac=region_row_min_run_frac,
            debug_dir=debug_dir,
            debug_prefix=debug_prefix
        )
    return out

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

    common_params = (
        config.out,
        config.ns,
        config.landscape,
        config.crop_region,
        tuple(config.region_bgr) if config.region_bgr else None,
        config.region_hsv_tol[0],
        config.region_hsv_tol[1],
        config.region_hsv_tol[2],
        config.region_margin,
        config.region_min_area_frac,
        config.region_close[0],
        config.region_close[1],
        None if str(config.region_row).lower() == "auto" else float(config.region_row),
        config.region_row_min_run,
        config.region_debug_dir,
        config.region_debug_prefix,
        config.verbose,
        config.sufix)

    if config.cpu > 1 and len(input_files) > 1:
        from concurrent.futures import ThreadPoolExecutor

        bar = tqdm(desc="Processing video files...\n", total=len(input_files), position=0)
        videos = {}
        with ThreadPoolExecutor(max_workers=config.cpu) as executor:
            for f in input_files:
                extractor_params = (os.path.join(config.vdata, f),) + common_params
                ex = executor.submit(_run_extractor,*extractor_params)
                ex.add_done_callback(lambda x: bar.update(1))
                videos[ex] = f

        for task in concurrent.futures.as_completed(videos):
            extracted_frames += task.result()
            analyzed_videos += 1
    else:
        for f in input_files:
            if config.verbose:
                print(f"Extracting from {f}...")

            new_imgs = _run_extractor(os.path.join(config.vdata, f), *common_params)
            extracted_frames += new_imgs
            analyzed_videos += 1

    print("Extracted {} frames from {} videos".format(extracted_frames, analyzed_videos))


def _run_extractor(video_path: str, dst_dir: str, nfps: int,
                   force_landscape:bool = True,
                   crop_region:bool = True,
                   region_bgr:Tuple[int, int, int, int] = None,
                   region_tol_h:int = 0,
                   region_tol_s:int = 0,
                   region_tol_v:int = 0,
                   region_margin_px:int=15,
                   region_min_area_frac:float = 0.25,
                   region_close_x_frac:float = 0.1,
                   region_close_y_frac:float = 0.1,
                   region_row_thresh:float | None = None,
                   region_row_min_run_frac:float = 0.2,
                   debug_dir:str | None = None,
                   debug_prefix:str | None = None,
                   verbosity: int = 0,
                   sufix: str = ''):
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
            raw_img = vid[j].asnumpy()
            proc = preprocess_frame(
                raw_img,
                force_landscape=force_landscape,
                crop_region=crop_region,
                region_bgr=tuple(region_bgr) if region_bgr else None,
                region_tol_h=region_tol_h,
                region_tol_s=region_tol_s,
                region_tol_v=region_tol_v,
                region_margin_px=region_margin_px,
                region_min_area_frac=region_min_area_frac,
                region_close_x_frac=region_close_x_frac,
                region_close_y_frac=region_close_y_frac,
                region_row_thresh=region_row_thresh,
                region_row_min_run_frac=region_row_min_run_frac,
                debug_dir=debug_dir,
                debug_prefix=f"{video_name}-{j}" if debug_prefix is None else debug_prefix,
            )
            cv2.imwrite(os.path.join(dst_dir, '{}-{}{}.jpg'.format(video_name, j, sufix)),
                        cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
            extracted += 1

        fcount += fps

    # vid.release()
    return extracted
