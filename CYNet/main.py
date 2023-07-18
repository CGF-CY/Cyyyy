import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.Load_CelebA import CelebAData
from torchvision import transforms
from networks.MyNet import MyNet
import torch.optim as optim
import argparse
from configs import parse





args = parse().parse_args()


def test_accuracy(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            probabilities = outputs.view(-1)
            # 将概率值转换为类别
            predicted = (probabilities > 0.5).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    model.train()  # 将模型设置回训练模式
    return accuracy


train_transform = transforms.Compose(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
test_transform = transforms.Compose(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

)
train_data = CelebAData(root_dir=args.data_dir, train=True, transforms=train_transform)
test_data = CelebAData(root_dir=args.data_dir, train=True, transforms=test_transform)
# data

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

# model
gpu_ids = [0, 1]
device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
model = MyNet()
model.to(device)

# optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

num_epochs = args.num_epochs

for epoch in range(num_epochs):
    print(f"Staring {epoch} time")
    running_loss = 0.0
    train_num = 0
    for data in train_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_num += 1

    # 计算训练损失并打印
    train_loss = running_loss / (train_num + 1)
    # 计算和打印测试精度
    test_acc = test_accuracy(model, test_loader, device)
    print(f"Epoch {epoch + 1}, Loss: {train_loss}, Test Accuracy: {test_acc}")

print("OK!!!!")










