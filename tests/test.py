from src import encoder
from PIL import Image
from torchvision import datasets, transforms


model = encoder.DenseNet(16)
image = Image.open('../../Dataset/Formula/formulae/val/0000030.png')
transform = transforms.Compose([
        transforms.Resize(512),
        transforms.Pad((0, 0, 0, 0), fill=0),
        transforms.CenterCrop((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
image = transform(image)
image = image.unsqueeze(0)
output = model(image)
print(output.size())
