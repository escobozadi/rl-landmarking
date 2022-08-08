import torch
import cv2
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class BaselineModel(nn.Module):

    def __init__(self, targets=8):
        super(BaselineModel, self).__init__()
        self.targets = targets  # number of total landmarks
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Pretrained Segmentation Model
        # self.backbone = deeplabv3_mobilenet_v3_large(pretrained_backbone=True)
        # self.backbone = self.backbone.backbone
        # layer_count = 0
        # for child in self.backbone.children():
        #     if layer_count <= 7:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     layer_count += 1
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:7]).to(self.device)
        # print(self.backbone)

        self.conv0 = nn.Conv2d(
            in_channels=3,
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

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(5, 5),
            padding=1).to(self.device)
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=(2, 2)).to(self.device)

        self.prelu0 = nn.PReLU().to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(
                in_features=16*14*14,
                out_features=256).to(
                self.device) for _ in range(
                self.targets + 1)])
        self.prelu3 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets + 1)])

        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256,
                out_features=128).to(
                self.device) for _ in range(
                self.targets + 1)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets + 1)])

        # Predict target (x,y)
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128,
                out_features=2).to(
                self.device) for _ in range(
                self.targets)])

        self.fcl = nn.Linear(in_features=128, out_features=targets).to(self.device)

        return

    @autocast()
    def forward(self, input):
        # Input:    batch x (3, 256, 256)
        # x = self.backbone(input)
        x = self.prelu0(self.conv0(input))  # N x (32, 254, 254)
        x = self.maxpool0(x)    # N x (32, 127, 127)
        x = self.prelu1(self.conv1(x))  # N x (32, 125, 125)
        x = self.maxpool1(x)    # N x (32, 31, 31)
        x = self.prelu2(self.conv2(x))  # N x (16, 29, 29)
        x = self.maxpool2(x)    # N x (16, 14, 14)
        conv_out = x.view(-1, 16*14*14)

        output = []
        # Landmark Detection Layers
        for i in range(self.targets):
            x = self.prelu3[i](self.fc1[i](conv_out))
            x = self.prelu4[i](self.fc2[i](x))
            x = self.fc3[i](x)
            output.append(x)
        output = torch.stack(output, dim=1)

        # Landmark Classification Layers
        y = self.prelu3[-1](self.fc1[-1](conv_out))
        y = self.prelu4[-1](self.fc2[-1](y))
        classification = self.fcl(y)

        return output, classification


########################################################################################################################
# Experimenting with model

if __name__ == '__main__':
    model = BaselineModel()
    import numpy as np
    path = "/Users/dianaescoboza/Documents/PycharmProjects/rl-landmark/rl-medical/src/data/train/0a0a5d3d-m527_a528_st533_se536_i21457_1_46_US_.png"

    image = cv2.imread(path)  # .transpose(2, 0, 1)
    x = round(0.4073333333333333 * 256)
    y = round(0.6685823754789273 * 256)
    # # image = cv2.copyMakeBorder(image, 0, 786 - image.shape[0], 0, 1136 - image.shape[1], cv2.BORDER_CONSTANT)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float() / 255
    a, b = model.forward(image.unsqueeze(0))
    print(a)
    print(b)
    # size = image.shape
    # image = image / 255
    # noise_img = image + np.random.rand(size[0], size[1], size[2]) * 0.5
    # cv2.circle(image1, (x, y), radius=2, color=255, thickness=-1)
    # cv2.circle(image2, (x, y), radius=2, color=255, thickness=-1)
    # cv2.circle(image3, (x, y), radius=2, color=255, thickness=-1)
    # imagescat = np.concatenate((image1, image2, image3), axis=1)
    # cv2.imshow("image resize", imagescat)
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
    # # images max size:
    # (786, 1136, 3)
    # Old model
    # output1 = []
    # for i in range(self.targets + 1):
    #     x = self.fc1[i](conv_out)
    #     output1.append(self.prelu2[i](x))
    # output1 = torch.stack(output1, dim=1)
    #
    # output2 = []
    # for i in range(self.targets + 1):
    #     x = self.fc2[i](output1[:, i])
    #     output2.append(self.prelu3[i](x))
    # output2 = torch.stack(output2, dim=1)
    #
    # output3 = []
    # for i in range(self.targets):
    #     x = self.fc3[i](output2[:, i])
    #     output3.append(self.sigmoid[i](x))
    # output3 = torch.stack(output3, dim=1)
    # classification = self.sigmoid[-1](self.fc(output2[:, -1]))
