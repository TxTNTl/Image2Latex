from encoder_decoder import *
from utils import *
from torch import optim
import time
import config


def train(train_image_dir):
    train_dataset = ParquetDataset(train_image_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = FormulaRecognitionModel(config.growth_rate, config.hidden_size, config.attention_dim,
                                    config.vocabulary_size)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for i in range(config.num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            if images.size(0) < 32:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{i} / {config.num_epochs - 1}, loss = {running_loss / len(train_loader)}")

    output_dir = config.model_output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    formatted_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    torch.save(model, os.path.join(output_dir, f'{formatted_time}.pth'))


def continue_training(train_image_dir):
    print("enter your model name including .pth")
    name = input()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(config.model_output_dir, name)
    if not os.path.exists(model_dir):
        print("wrong model name")
    else:
        train_dataset = ParquetDataset(train_image_dir)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = torch.load(model_dir, weights_only=False, map_location=device)
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for i in range(config.num_continue_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                if images.size(0) < 32:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, labels)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"{i} / {config.num_epochs - 1}, loss = {running_loss / len(train_loader)}")
        remain_name, _ = model_dir.split('.pt')
        torch.save(model, os.path.join(remain_name, f'(1).pth'))


if __name__ == '__main__':
    train(config.train_image_dir)
