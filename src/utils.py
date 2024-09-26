import os
import torch
import io
import numpy as np
import sympy as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import *


class FormulaDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=transforms.Compose([
        transforms.Resize(512),
        transforms.Pad((0, 0, 0, 0), fill=0),
        transforms.CenterCrop((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])):
        self.img_dir = image_dir
        self.labels = self._load_labels(label_file)
        self.transform = transform
        self.dict = generate_dictionary(self.labels)
        self.max_length = basic_dict['max_length']
        self.image_files = sorted(os.listdir(self.img_dir))

    def _load_labels(self, label_file):
        # 读取所有的LaTeX公式，逐行存储到列表中
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        return [label.strip() for label in labels]  # 去除每行末尾的换行符等空白字符

    def one_hot_encode(self, label):
        # create one-hot code matrix
        one_hot = np.zeros((self.max_length, len(self.dict)))   # 200 696
        tokens = label.split()
        ls = []
        for token in tokens:
            token = token.strip()
            if len(token) > 0:
                ls.append(token)
        for t, token in enumerate(ls):
            if t >= self.max_length:
                break
            index = self.dict.get(token, -1)
            if index != -1:
                one_hot[t, index] = 1
        one_hot[len(ls), 695] = 1
        return one_hot

    def __len__(self):
        # 返回图片文件的数量，假设图片文件名是从 0 开始连续编号的
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.one_hot_encode(label)


class ParquetDataset(Dataset):
    def __init__(self, parquet_dir, transform=transforms.Compose([
        transforms.Resize(512),
        transforms.Pad((0, 0, 0, 0), fill=0),
        transforms.CenterCrop((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])):
        self.data = pd.read_parquet(parquet_dir)
        self.transform = transform
        self.max_length = basic_dict['max_length']
        self.data['image'] = self.data['image'].apply(lambda x: self.bytes_to_image(x['bytes']))
        self.data['image'] = self.data['image'].apply(lambda img: transform(img))
        self.images = self.data['image'].tolist()
        self.labels = self.data['text'].tolist()
        self.dict = generate_dictionary(self.labels)

    def bytes_to_image(self, image_bytes):
        return Image.open(io.BytesIO(image_bytes))

    def one_hot_encode(self, label):
        dict_size = len(self.dict)
        one_hot = torch.zeros(self.max_length, dict_size)
        tokens = label.split()
        ls = []
        for token in tokens:
            token = token.strip()
            if len(token) > 0:
                ls.append(token)
        for t, token in enumerate(ls):
            if t >= self.max_length:
                break
            index = self.dict.get(token, -1)
            if index != -1:
                one_hot[t, index] = 1
        one_hot[len(ls), dict_size] = 1
        return one_hot

    def __len__(self):
        # 返回图片文件的数量，假设图片文件名是从 0 开始连续编号的
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.one_hot_encode(self.labels[idx])


def one_hot_decode(output_seq, dictionary):
    predicted_indices = torch.argmax(output_seq, dim=2)  # (batch_size=1, max_length, vocab_size)
    dict_reversed = reverse_dictionary(dictionary)
    latex_seq = [dict_reversed[idx] for idx in predicted_indices[0].cpu().numpy()]

    if '<eol>' in latex_seq:
        eos_index = latex_seq.index('<eol>')
        latex_seq = latex_seq[:eos_index]
    final_latex_sequence = ''.join(latex_seq)

    return final_latex_sequence


def generate_dictionary(labels):
    dict = {}
    count = 0
    for label in labels:
        label = str(label)
        tokens = label.split()
        for token in tokens:
            token = token.strip()
            if len(token) > 0 and token not in dict.keys():
                dict[token] = count
                count += 1
    dict['<eol>'] = count
    print(dict)
    return dict


def reverse_dictionary(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[value] = key
    return new_dict


def acquire_max_length(labels):
    max_length = 0
    for label in labels:
        tokens = label.split()
        count = 0
        for token in tokens:
            token = token.strip()
            if len(token) > 0:
                count += 1
        max_length = max(max_length, count + 1)     # +1是<eol>
    return max_length
