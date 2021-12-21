import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
import copy
import math
import torch.utils.model_zoo as model_zoo

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
data_dir = "./datasets"
model_save = "./resnet.pth"
num_classes = 5
batch_size = 32
num_epochs = 30
input_size = 224
lr = 0.01
momentum = 0.9
grad = True         # 是否计算梯度
pretrain = True     # 是否使用预训练集
training = True     # 是否训练
testing = True      # 是否测试

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 预训练模型
pre_model = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ResNet的公共部分
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(pre_model))
    return model


# 测试
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = loss_func(outputs, labels)

        _, predicts = torch.max(outputs, 1)

        loss_val += loss.item() * images.size(0)
        corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()
        # print(labels)
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)

    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))

    return test_acc


# 训练
def train(model, train_loader, test_loader, loss_func, optimizer, num_epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            _, predicts = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item() * images.size(0)
            corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()

        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)

        print("round{}:\nTrain Loss: {}, Train Acc: {}".format(epoch, train_loss, train_acc))
        test_acc = test(model, test_loader, loss_func)
        # 选择模型记录
        if (best_val_acc < test_acc):
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model


# 梯度
def set_parameters_require_grad(model, grad):
    if (grad):
        for parameter in model.parameters():
            parameter.requires_grad = False


def init_model(num_classes, grad, pretrained):
    model = resnet18(pretrained=pretrain)
    set_parameters_require_grad(model, grad)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_datasets(data_dir, input_size, is_train_data):
    if (is_train_data):
        images = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                      transforms.Compose([
                                          transforms.RandomResizedCrop(input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                      ]))
    else:
        images = datasets.ImageFolder(os.path.join(data_dir, "test"),
                                      transforms.Compose([
                                          transforms.Resize(input_size),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor()
                                      ]))
    return images


# 更新模型
def get_require_updated_params(model, grad):
    if (grad):
        require_update_params = []
        for param in model.parameters():
            if (param.requires_grad):
                require_update_params.append(param)
        return require_update_params
    else:
        return model.parameters()


def main():
    train_images = get_datasets(data_dir, input_size, is_train_data=True)
    test_images = get_datasets(data_dir, input_size, is_train_data=False)

    train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size)

    model = init_model(num_classes, grad, pretrain)
    model = model.to(device)

    require_update_params = get_require_updated_params(model, grad)

    optimizer = torch.optim.SGD(require_update_params, lr=lr, momentum=momentum)
    loss_func = nn.CrossEntropyLoss()

    if (training):
        model = train(model, train_loader, test_loader, loss_func, optimizer, num_epochs)
        torch.save(model.state_dict(), model_save)
    if (testing):
        model.load_state_dict(torch.load(model_save))
        acc = test(model, test_loader, loss_func)
        print("Best Test Acc: {}".format(acc))


if __name__ == "__main__":
    main()
