# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
import torch.nn as nn
from IPython.display import Image
import os

train_data_dir = 'archive/data/data/train'
train_data = os.listdir(train_data_dir)

