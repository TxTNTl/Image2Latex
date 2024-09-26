growth_rate = 16
hidden_size = 512
attention_dim = 512
vocabulary_size = 556
num_epochs = 25
model_output_dir = '../models/'
dictionary_dir = ''
max_length = 151
train_image_dir = '../../Dataset/full/train-00000-of-00001.parquet'
train_label_dir = '../../Dataset/full/train-00000-of-00001.parquet'
test_image_dir = '../../Dataset/full/test-00000-of-00001.parquet'
test_label_dir = '../../Dataset/full/test-00000-of-00001.parquet'


basic_dict = {
    'growth_rate': 16,
    'hidden_size': 512,
    'attention_dim': 512,
    'vocab_size': 696,
    'num_epochs': 10,
    'output_dir': '../models/',
    'dictionary_dir': './vocabulary.txt',
    'max_length': 2000,
}

train_dict = {
    'image_dir': '../../Dataset/Formula/formulae/train/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}

test_dict = {
    'image_dir': '../../Dataset/Formula/formulae/test/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}
