"""Resume the same locked 120 clips only; never a full-cache entry."""
import argparse
import torch
import cv2
from .cache import build_cache
from .support import enable_guard


def main():
    enable_guard();torch.set_num_threads(4);cv2.setNumThreads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-after-new-frames",type=int,help="Pause after reaching this threshold at a complete shard boundary (at most 7 extra frames)")
    args=parser.parse_args()
    if args.pause_after_new_frames is not None and args.pause_after_new_frames<1:raise ValueError("Positive pause threshold required")
    build_cache(max_new_frames=args.pause_after_new_frames)


if __name__=="__main__":main()
