from utils import ParquetDataset
import config


dataset = ParquetDataset(config.train_image_dir)
dictionary = {}
count = 0
for label in dataset.labels:
    label = str(label)
    tokens = label.split()
    for token in tokens:
        token = token.strip()
        if len(token) > 0 and token not in dictionary.keys():
            dictionary[token] = count
            count += 1
dictionary['<eol>'] = count
config.dictionary = dictionary

print(dictionary)
with open("dictionary.txt", 'w') as f:
    for key, value in dictionary.items():
        f.write(f"{key} {value}\n")
print(len(dictionary))


