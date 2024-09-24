from encoder_decoder import *
from utils import *


print("enter your model name including .pth")
name = input()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = os.path.join(basic_dict['output_dir'], name)
if not os.path.exists(model_dir):
    print("wrong model name")
else:
    model = torch.load(model_dir, weights_only=False, map_location=device)

    model.eval()
    model.to(device)

    train_dataset = FormulaDataset(test_dict['image_dir'], train_dict['label_dir'])
    train_loader = DataLoader(test_dict, batch_size=32, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, hidden = model(images)
            for output, label in zip(outputs, labels):
                predicted = one_hot_decode(output)
                latex = one_hot_decode(label)
                if predicted == latex:
                    print(f"预测为{predicted}，实际为{label}，结果正确")
                    correct += 1
                else:
                    print(f"预测为{predicted}，实际为{label}，结果错误")
                total += 1
    print("测试完成")
    print(f"预测成功数量为{correct}，总预测数为{total}预测准确率为{correct / total}")
