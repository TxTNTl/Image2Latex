import utils
import torch
from config import *
from torch.utils.data import DataLoader


dataset = utils.ParquetDataset(train_image_dir)
print(len(utils.generate_dictionary(dataset.labels).items()))
print(utils.acquire_max_length(dataset.labels))