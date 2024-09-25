from encoder_decoder import *
from utils import *
from torch import optim
import time

model = FormulaRecognitionModel()

train_dataset = FormulaDataset(train_dict['image_dir'], train_dict['label_dir'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1, weight_decay=1e-4, eps=1e-6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for i in range(basic_dict['num_epochs']):
    running_loss = 0.0
    for images, labels in train_loader:
        if images.size(0) < 32:
            continue
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, labels)
        print(outputs.requires_grad)
        print(labels.requires_grad)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("1")
    print(f"{i} / {basic_dict['num_epochs'] - 1}, loss = {running_loss / len(train_loader)}")

output_dir = basic_dict['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
torch.save(model, os.path.join(output_dir, f'{formatted_time}.pth'))
