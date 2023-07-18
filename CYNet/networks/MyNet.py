import torch.nn as nn
import torch
import torchvision
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.pretrain_model=torchvision.models.resnet50(pretrained=False)
        weights_path="C:/Users/Administrator/PycharmProjects/CYNet/resnet50-0676ba61.pth"
        self.pretrain_model.load_state_dict(torch.load(weights_path))
        self.fc1=nn.Linear(self.pretrain_model.fc.out_features,500)
        self.fc2=nn.Linear(500,2)

    def forward(self,x):
        x=self.pretrain_model(x)
        x=nn.ReLU(x)
        x=self.fc1(x)
        x=nn.ReLU(x)
        x=self.fc2(x)
        x=nn.Sigmoid(x)
        return x