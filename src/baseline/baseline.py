import torch
import cv2
import os
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from torchvision.models.segmentation import \
    deeplabv3_mobilenet_v3_large


class BaselineModel(nn.Module):

    def __init__(self, targets=8):
        super(BaselineModel, self).__init__()
        self.targets = targets

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Pretrained Segmentation Model
        self.backbone = deeplabv3_mobilenet_v3_large(pretrained_backbone=True).backbone
        layer_count = 0
        for child in self.backbone.children():
            if layer_count <= 7:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            layer_count += 1

        self.backbone = nn.Sequential(*list(self.backbone.children())[:7]).to(self.device)
        print(self.backbone)

        self.conv0 = nn.Conv2d(
            in_channels=40,
            out_channels=32,
            kernel_size=(5, 5),
            padding=1).to(self.device)  # (32,61-2,59)
        self.maxpool0 = nn.MaxPool2d(
            kernel_size=(2, 2)).to(self.device)  # (32,(59-2)/2 +1,29)

        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            padding=1).to(self.device)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=(4, 4)).to(self.device)  # (32, 23, 34)

        self.prelu0 = nn.PReLU().to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(
                in_features=32*11*17,
                out_features=256).to(
                self.device) for _ in range(
                self.targets)])
        self.prelu2 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets)])

        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256,
                out_features=128).to(
                self.device) for _ in range(
                self.targets)])
        self.prelu3 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets)])

        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128,
                out_features=targets).to(
                self.device) for _ in range(
                self.targets)])

        return

    def forward(self, input):
        output = self.backbone.forward(input)
        output = self.conv0(output)
        output = self.prelu0(output)
        output = self.maxpool0(output)
        output = self.conv1(output)
        output = self.prelu1(output)
        output = self.maxpool1(output)
        output = output.view(-1, 32*11*17)
        # for i in range(self.targets):
        #     output = self.fc1(output)
        output = self.fc1[0](output)
        output = self.prelu2[0](output)
        output = self.fc2[0](output)
        output = self.prelu3[0](output)
        output = self.fc3[0](output)

        return output


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
    out = model.forward(image.unsqueeze(0))
    print(out.shape)

    # cv2.circle(image, (x, y), radius=5, color=255, thickness=-1)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    # width = 0
    # height = 0
    # max_width = []
    # max_height = []
    # files = [f for f in os.listdir(path) if not f.startswith('.')]
    # for image in files:
    #     im = cv2.imread(path + image)
    #     size = im.shape
    #     if size[0] > width:
    #         width = size[0]
    #         max_width = size
    #     if size[1] > height:
    #         height = size[1]
    #         max_height = size
    # print(width)
    # print(max_width)
    # print(height)
    # print(max_height)

    # If a sequence of length 4 is provided
    # this is the padding for the left, top, right and bottom borders
    # max size: 786
    # (786, 1136, 3)
    # 1136
    # (786, 1136, 3)

    # Original image size: torch.Size([3, 786, 1136])
    # torch.Size([1, 80, 50, 71])

    # Original image size: (3, 500, 720)
    # torch.Size([1, 40, 63, 90])
