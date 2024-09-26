import train
import test
import config


if __name__ == '__main__':
    train.train(config.feasible_train_dir)
    test.test(config.feasible_test_dir)