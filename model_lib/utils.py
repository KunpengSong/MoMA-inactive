import argparse
import os
import torch
import pdb
import random 
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model_path",type=str,default="liuhaotian/llava-v1.5-7b",help="The llava base model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args

