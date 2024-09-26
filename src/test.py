from utils import *


def test(test_image_dir):
    print("enter your model name including .pth")
    name = input()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(config.model_output_dir, name)
    if not os.path.exists(model_dir):
        print("wrong model name")
    else:
        test_dataset = ParquetDataset(test_image_dir)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        model = torch.load(model_dir, weights_only=False, map_location=device)

        model.eval()
        model.to(device)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs= model(images)
                for output, label in zip(outputs, labels):
                    predicted = test_dataset.one_hot_decode(output)
                    latex = test_dataset.one_hot_decode(label)
                    if predicted == latex:
                        print(f"预测为{predicted}\n实际为{label}，结果正确")
                        correct += 1
                    else:
                        print(f"预测为{predicted}\n实际为{latex}，结果错误")
                    total += 1
        print("测试完成")
        print(f"预测成功数量为{correct}，总预测数为{total}预测准确率为{correct / total}")


if __name__ == '__main__':
    test(config.test_image_dir)
