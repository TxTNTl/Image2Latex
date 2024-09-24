basic_dict = {
    'growth_rate': 16,
    'hidden_size': 512,
    'attention_dim': 512,
    'vocab_size': 113,
    'num_epochs': 10,
    'output_dir': '../models/',
    'dictionary_dir': './dictionary.txt',
    'max_length': 100,
}

train_dict = {
    'image_dir': '../../Dataset/Formula/formulae/train/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}

test_dict = {
    'image_dir': '../../Dataset/Formula/formulae/test/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}
