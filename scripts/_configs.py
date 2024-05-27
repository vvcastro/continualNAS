import warnings
import sys
import os

warnings.simplefilter("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CIFAR10_DATA_DIR = "assets/datasets/cifar10"
OFA_MODEL_PATH = "assets/supernets/ofa_mbv3_d234_e346_k357_w1.0"
