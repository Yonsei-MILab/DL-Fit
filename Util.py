import os
import sys
import matplotlib as mpl
import matplotlib_inline as mpli
from matplotlib import pyplot as plt
import torch
import random
import numpy as np

def configure(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    sys.path.insert(0, "../libs/dgibbs")
    sys.path.insert(0, "../libs/dgibbs-torch")
    mpl.rc("image", interpolation="none")
    mpl.rc("figure", dpi=300)
    mpli.backend_inline.set_matplotlib_formats("retina")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
