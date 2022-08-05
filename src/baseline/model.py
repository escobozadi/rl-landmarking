import torch
import cv2
import os
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3

deeplabv3_mobilenet_v3_large()

class BaselineModel(nn.Module):

    def __init__(self, targets=8):
        super(BaselineModel, self).__init__()
        self.targets = targets
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Pretrained Segmentation Model
        self.backbone = deeplabv3_mobilenet_v3_large(pretrained_backbone=True)
        self.backbone = self.backbone.backbone
        layer_count = 0
        for child in self.backbone.children():
            if layer_count <= 7:
                for param in child.parameters():
                    param.requires_grad = False
            layer_count += 1

        self.backbone = nn.Sequential(*list(self.backbone.children())[:7]).to(self.device)
        # print(self.backbone)

        self.conv0 = nn.Conv2d(
            in_channels=40,
            out_channels=32,
            kernel_size=(5, 5),
            padding=1).to(self.device)
        self.maxpool0 = nn.MaxPool2d(
            kernel_size=(2, 2)).to(self.device)

        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            padding=1).to(self.device)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=(4, 4)).to(self.device)

        self.prelu0 = nn.PReLU().to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(
                in_features=32*11*17,
                out_features=256).to(
                self.device) for _ in range(
                self.targets + 1)])
        self.prelu2 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets + 1)])

        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256,
                out_features=128).to(
                self.device) for _ in range(
                self.targets + 1)])
        self.prelu3 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets + 1)])

        # Predict box center(x,y), height, width
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128,
                out_features=4).to(
                self.device) for _ in range(
                self.targets)])
        self.sigmoid = nn.ModuleList(
            [nn.Sigmoid().to(self.device) for _ in range(self.targets + 1)])

        self.fc = nn.Linear(in_features=128, out_features=targets).to(self.device)
        # self.softmax = nn.Softmax()

        return

    def forward(self, input):
        input1 = input / 255.0

        #x = self.backbone.forward(input1)
        x = self.backbone(input1)
        x = self.conv0(x)
        x = self.prelu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)
        conv_out = x.view(-1, 32*11*17)

        output1 = []
        for i in range(self.targets + 1):
            x = self.fc1[i](conv_out)
            output1.append(self.prelu2[i](x))
        output1 = torch.stack(output1, dim=1)

        output2 = []
        for i in range(self.targets + 1):
            x = self.fc2[i](output1[:, i])
            output2.append(self.prelu3[i](x))
        output2 = torch.stack(output2, dim=1)

        output3 = []
        for i in range(self.targets):
            x = self.fc3[i](output2[:, i])
            output3.append(self.sigmoid[i](x))
        output3 = torch.stack(output3, dim=1)
        classification = self.sigmoid[-1](self.fc(output2[:, -1]))

        output3 = output3.cpu()
        classification = classification.cpu()
        return output3, classification


if __name__ == '__main__':
    model = BaselineModel()
    path = "/Users/dianaescoboza/Documents/PycharmProjects/rl-landmark/rl-medical/src/data/images/0a0a5d3d-m527_a528_st533_se536_i21457_1_46_US_.png"

    # # image = cv2.resize(cv2.imread(path), (512, 512)).transpose(2, 0, 1)
    # image = cv2.imread(path).transpose(2, 0, 1)
    # image = image.astype(np.uint8) / 255

    image = cv2.imread(path)  # .transpose(2, 0, 1)
    x = round(0.4073333333333333 * image.shape[1])
    y = round(0.6685823754789273 * image.shape[0])
    print(image.shape)
    image = cv2.copyMakeBorder(image, 0, 786 - image.shape[0], 0, 1136 - image.shape[1], cv2.BORDER_CONSTANT)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float() / 255
    print("Original image size: {}".format(image.shape))
    out, c = model.forward(image.unsqueeze(0))
    print("Landmarks location: ")
    print(out.shape)
    print(out)
    print("Landmarks: ")
    print(c.shape)
    print(c)

    cv2.circle(image, (x, y), radius=5, color=255, thickness=-1)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    width = 0
    height = 0
    max_width = []
    max_height = []
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    for image in files:
        im = cv2.imread(path + image)
        size = im.shape
        if size[0] > width:
            width = size[0]
            max_width = size
        if size[1] > height:
            height = size[1]
            max_height = size
    print(width)
    print(max_width)
    print(height)
    print(max_height)

    # If a sequence of length 4 is provided
    # this is the padding for the left, top, right and bottom borders
    # max size: 786
    # (786, 1136, 3)
    # 1136
    # (786, 1136, 3)
    #
    # Original image size: torch.Size([3, 786, 1136])
    # torch.Size([1, 80, 50, 71])
    #
    # Original image size: (3, 500, 720)
    # torch.Size([1, 40, 63, 90])
