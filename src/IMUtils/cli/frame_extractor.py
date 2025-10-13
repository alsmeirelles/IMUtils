"""
CLI shim for the frame extraction pipeline.

This wraps the existing extractor so users can run:
    imutils-frames -vdata <videos_dir> -out <frames_dir> -ns 3 -cpu 4

It reuses your functions and keeps the same flags for zero friction.
"""
import os
import sys
import argparse

# Import your existing functions (assumes moved to src/IMUtils/FrameExtractor.py)
from IMUtils.FrameExtractor import (
    run_frame_extraction,
    run_annotated_frame_extraction,
)

def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser mirroring the original script."""
    p = argparse.ArgumentParser(description="Extract frames from video files.")
    p.add_argument('--vdata', dest='vdata', type=str, default='',
                   help='Path to folder containing video files.', required=False)
    p.add_argument('--fdata', dest='fdata', type=str, default='',
                   help='Path to a specific video file.', required=False)
    p.add_argument('--frames', dest='frames', type=str, default='',
                   help='Path to dir with .txt files listing frame numbers to extract.', required=False)
    p.add_argument('--out', dest='out', type=str, default='',
                   help='Save frames here.', required=True)
    p.add_argument('--nb', dest='nb', type=int, default=1,
                   help='With annotations, extract nb frames before and after each frame.')
    p.add_argument('--ns', dest='ns', type=int, default=3,
                   help='Extract this many frames per second (0 for all).')
    p.add_argument('--cpu', dest='cpu', type=int, default=1,
                   help='Number of worker threads for parallel extraction.')
    p.add_argument('--v', action='count', default=0, dest='verbose',
                   help="Increase verbosity; use -v, -vv, etc.")
    p.add_argument('--sf', dest='sufix', type=str, default='',
                   help='Append suffix to output frames as _suffix.', required=False)
    return p

def _tuple_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(","))

def _tuple_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(","))

def _add_preprocess_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add preprocessing CLI arguments."""
    g = parser.add_argument_group("Preprocess")
    g.add_argument("--landscape", action="store_true",
                   help="Rotate to landscape when frame is portrait.")
    g.add_argument("--no-landscape", dest="landscape", action="store_false")
    g.set_defaults(landscape=True)

    g.add_argument("--crop-region", action="store_true",
                   help="Enable dominant color region cropping.")
    g.add_argument("--region-bgr", type=_tuple_ints, default=None,
                   help="BGR triplet, e.g. 255,100,20. Omit for auto detection.")
    g.add_argument("--region-hsv-tol", type=_tuple_ints, default=(16,80,80),
                   help="HSV tolerances (H,S,V). Example: 16,80,80.")
    g.add_argument("--region-margin", type=int, default=50,
                   help="Margin (px) to expand around detected region.")
    g.add_argument("--region-min-area-frac", type=float, default=0.08,
                   help="Reject small contours (< fraction of image).")
    g.add_argument("--region-close", type=_tuple_floats, default=(1/18, 1/120),
                   help="Morph close kernel fractions (x_frac,y_frac).")
    g.add_argument("--region-row", default="auto",
                   help="Row ratio threshold (0..1) or 'auto' for adaptive.")
    g.add_argument("--region-row-min-run", type=float, default=0.35,
                   help="Minimum vertical run (fraction of H) to accept fallback band.")
    g.add_argument("--region-debug-dir", default=None,
                   help="If set, saves mask/closed images and row_ratio.csv for tuning.")
    g.add_argument("--region-debug-prefix", default=None,
                   help="File name prefixes for tuning.")

def main() -> None:
    """Entry point called by the `imutils-frames` console script."""
    parser = _build_arg_parser()
    _add_preprocess_cli_args(parser)

    config, _ = parser.parse_known_args()

    if not config.vdata and not config.fdata:
        print("You should define a path to a dir (vdata) or file (fdata)")
        sys.exit(1)

    if config.out and not os.path.isdir(config.out):
        os.makedirs(config.out, exist_ok=True)

    if config.frames:
        run_annotated_frame_extraction(config)
    else:
        run_frame_extraction(config)

if __name__ == "__main__":
    main()
