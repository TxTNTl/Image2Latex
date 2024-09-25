basic_dict = {
    'growth_rate': 16,
    'hidden_size': 512,
    'attention_dim': 512,
    'vocab_size': 696,
    'num_epochs': 10,
    'output_dir': '../models/',
    'dictionary_dir': './vocabulary.txt',
    'max_length': 700,
}

train_dict = {
    'image_dir': '../../Dataset/Formula/formulae/train/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}

test_dict = {
    'image_dir': '../../Dataset/Formula/formulae/test/',
    'label_dir': '../../Dataset/Formula/Math.txt',
}
