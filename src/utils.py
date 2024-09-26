import os
import torch
import io
import config
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ParquetDataset(Dataset):
    def __init__(self, parquet_dir, transform=transforms.Compose([
        transforms.Resize(128),
        transforms.Pad((0, 0, 0, 0), fill=0),
        transforms.CenterCrop((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])):
        self.data = pd.read_parquet(parquet_dir)
        self.transform = transform
        self.data['image'] = self.data['image'].apply(lambda x: self.bytes_to_image(x['bytes']))
        self.data['image'] = self.data['image'].apply(lambda img: transform(img))
        self.images = self.data['image'].tolist()
        self.labels = self.data['text'].tolist()
        self.dictionary = generate_dictionary()
        self.max_length = config.max_length

    def bytes_to_image(self, image_bytes):
        return Image.open(io.BytesIO(image_bytes))

    def one_hot_encode(self, label):
        dict_size = len(self.dictionary)
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
            index = self.dictionary.get(token, -1)
            if index != -1:
                one_hot[t, index] = 1
        one_hot[len(ls), dict_size - 1] = 1
        return one_hot

    def one_hot_decode(self, output_seq):
        predicted_indices = torch.argmax(output_seq, dim=1)
        dict_reversed = reverse_dictionary(self.dictionary)
        latex_seq = [dict_reversed[int(idx)] for idx in predicted_indices]

        if '<eol>' in latex_seq:
            eos_index = latex_seq.index('<eol>')
            latex_seq = latex_seq[:eos_index]
        final_latex_sequence = ' '.join(latex_seq)

        return final_latex_sequence

    def __len__(self):
        # 返回图片文件的数量，假设图片文件名是从 0 开始连续编号的
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.one_hot_encode(self.labels[idx])


def generate_dictionary():
    dictionary = {}
    with open(config.dictionary_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(' ')
            dictionary[key] = int(value)
    return dictionary


def reverse_dictionary(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[int(value)] = key
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
    config.max_length = max_length
    return max_length
