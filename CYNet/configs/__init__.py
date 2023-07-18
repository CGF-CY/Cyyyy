import argparse
import os
import torch


"""
    一些如文件名之类的参数
    
"""

def parse():
    parper = argparse.ArgumentParser()
    parper.add_argument('--data_dir', type=str, default="F:/CelebA_Spoof", help='Your data dir')
    parper.add_argument('--batch_size', type=int, default=1, help='The number of input images')
    parper.add_argument('--base_lr', type=float, default=0.00001, help='The base learn rate of model')
    parper.add_argument('--num_epochs', type=int, default=20, help='Number of times to train the model')
    return parper


