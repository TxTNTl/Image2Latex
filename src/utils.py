import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import basic_dict
import numpy as np


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
        self.dict, _ = load_dict()
        self.max_length = basic_dict['max_length']
        self.image_files = sorted(os.listdir(self.img_dir))

    def _load_labels(self, label_file):
        # 读取所有的LaTeX公式，逐行存储到列表中
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        return [label.strip() for label in labels]  # 去除每行末尾的换行符等空白字符

    def one_hot_encode(self, label):
        # create one-hot code matrix
        one_hot = np.zeros((self.max_length, len(self.dict))) # 100 113
        for t, char in enumerate(label):
            if t >= self.max_length:
                break
            index = self.dict.get(char, -1)
            if index != -1:
                one_hot[t, index] = 1
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


def one_hot_decode(output_seq):
    predicted_indices = torch.argmax(output_seq, dim=2)  # (batch_size=1, max_length)
    _, dict_reversed = load_dict()
    latex_seq = [dict_reversed[idx] for idx in predicted_indices[0].cpu().numpy()]

    if '<eol>' in latex_seq:
        eos_index = latex_seq.index('<eol>')
        latex_seq = latex_seq[:eos_index]
    final_latex_sequence = ''.join(latex_seq)

    return final_latex_sequence


def load_dict():
    with open(basic_dict['dictionary_dir'], 'r') as f:
        lines = f.readlines()
        dict = {}
        for line in lines:
            word = line.strip().split()
            dict[word[0]] = int(word[1])
        dict_reversed = {}
        for key, value in dict.items():
            dict_reversed[value] = key
        # dict = {word, number}
        # dict_reversed = {number, word}
        return dict, dict_reversed
