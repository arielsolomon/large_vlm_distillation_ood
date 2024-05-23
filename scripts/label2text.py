import os
import shutil
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/cifar10/images/'

cls_list = os.listdir(root)
for cls in cls_list:

    with open(root[:-7]+'label2text.txt', 'a') as f:
        f.write(cls[2:]+' '+cls[0]+' '+cls[2:]+'\n')
