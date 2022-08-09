import torch
import cv2
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class BaselineModel(nn.Module):

    def __init__(self, targets=3, labelpred=False):
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
                self.targets)])
        self.prelu3 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets)])

        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256,
                out_features=128).to(
                self.device) for _ in range(
                self.targets)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.targets)])

        # Predict target (x,y)
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128,
                out_features=2).to(
                self.device) for _ in range(
                self.targets)])

        # Class prediction
        self.labelpred = labelpred
        if labelpred:
            self.labelfc1 = nn.Linear(in_features=16 * 14 * 14, out_features=256).to(self.device)
            self.labelprelu1 = nn.PReLU().to(self.device)
            self.labelfc2 = nn.Linear(in_features=256, out_features=128).to(self.device)
            self.labelprelu2 = nn.PReLU().to(self.device)
            self.labelout = nn.Linear(in_features=128, out_features=targets).to(self.device)

        return

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

        if self.labelpred:
            # Landmark Classification Layers
            y = self.labelprelu1(self.labelfc1(conv_out))
            y = self.labelprelu2(self.labelfc2(y))
            classification = torch.abs(self.labelout(y))
            return output, classification

        return output


class BaselineLogs(object):
    def __init__(self):
        pass

    def saveBaseline(self, path):
        log = path + "/logs.txt"
        train_loss = {}
        train_dists = {}
        val_loss = {}
        val_dists = {}
        tagent_losses = {}
        vagent_losses = {}

        type = "train"
        epoch = 0
        with open(log) as t:
            lines = [x.strip().split(" ") for x in list(t) if x]
        for line in lines:
            if len(line) <= 1:
                continue
            if line[0] == "EPOCH":
                epoch = int(line[1])
                if epoch not in train_loss:
                    train_loss[epoch] = []
                    train_dists[epoch] = []
                    tagent_losses[epoch] = {}
                else:
                    val_loss[epoch] = []
                    val_dists[epoch] = []
                    vagent_losses[epoch] = {}
                continue
            elif line[1] == "Total":
                if line[0] == "Train":
                    train_loss[epoch].append(line[3])
                    type = "train"
                elif line[0] == "Validation":
                    val_loss[epoch].append(line[3])
                    type = "val"
                continue
            elif line[0] == "Target":
                if type == "train":
                    train_loss[epoch].append(line[3][:-1])
                    train_loss[epoch].append(line[6])
                else:
                    val_loss[epoch].append(line[3][:-1])
                    val_loss[epoch].append(line[6])
                continue
            elif line[0] == "Agent" and line[2] == "Dist":
                if type == "train":
                    tagent_losses[epoch][line[1]] = [line[4], line[8]]
                else:
                    vagent_losses[epoch][line[1]] = [line[4], line[8]]
                continue
            elif line[0] == "Agent":
                if type == "train":
                    train_dists[epoch].append(line[2])
                    train_dists[epoch].append(line[6])
                    train_dists[epoch].append(line[10])
                else:
                    val_dists[epoch].append(line[2])
                    val_dists[epoch].append(line[6])
                    val_dists[epoch].append(line[10])
                continue

        with open(path + "/train-dists.json", "w") as file:
            json.dump(train_dists, file)

        with open(path + "/val-dists.json", "w") as file:
            json.dump(val_dists, file)

        with open(path + "/train-loss.json", "w") as file:
            json.dump(train_loss, file)

        with open(path + "/val-loss.json", "w") as file:
            json.dump(val_loss, file)

        with open(path + "/agent-trainloss.json", "w") as file:
            json.dump(tagent_losses, file)

        with open(path + "/agent-valloss.json", "w") as file:
            json.dump(vagent_losses, file)

        return

    def plot_baseline(self, path):
        train_dists = json.load(open(path + "/train-dists.json", "r"))
        val_dists = json.load(open(path + "/val-dists.json", "r"))
        train_loss = json.load(open(path + "/train-loss.json", "r"))
        val_loss = json.load(open(path + "/val-loss.json", "r"))
        agent_loss = json.load(open(path + "/agent-trainloss.json", "r"))
        agent0 = []
        for d in list(agent_loss.values()):
            agent0.append(float(d['0'][0]))
        print(agent_loss.values())
        # agent0 = np.asarray(list(agent_loss.values())).astype(float)[:, 0]
        print(agent0)

        epochs = np.asarray(list(train_loss.keys())).astype(int)
        agent = np.zeros((3, len(epochs)))
        vagent = np.zeros((3, len(epochs)))
        tdists = np.asarray(list(train_dists.values())).astype(float)
        vdists = np.asarray(list(val_dists.values())).astype(float)
        for i in range(3):
            agent[i] = tdists[:, i]
            vagent[i] = vdists[:, i]
        tloss = np.asarray(list(train_loss.values())).astype(float)
        vloss = np.asarray(list(val_loss.values())).astype(float)

        colors = ["k-", "c-", "b-", "g-", "r-", "m-", "y-", "k-"]
        plt.subplot(2, 3, 1)
        plt.plot(epochs, tloss[:, 0], "g-", label="Mean Total Loss")
        # plt.plot(epochs, vloss[:, 0], "r-", label="Val Total Loss")
        plt.title("Total Mean Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 2)
        plt.plot(epochs, tloss[:, 1], "g-", label="Train Mean Distance Loss")
        # plt.plot(epochs, vloss[:, 1], "r-", label="Val Mean Distance Loss")
        plt.title("Distance to Target Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 4)
        plt.plot(epochs, tloss[:, 2], "g-", label="Mean Class Loss")
        # plt.plot(epochs, vloss[:, 2], "r-", label="Val Class Loss")
        plt.title("Landmark Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 5)
        plt.plot(epochs, agent0, "g-", label="Agent 0 Distance Loss")
        plt.title("Agent 0")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 3)
        for i in range(3):
            plt.plot(epochs, agent[i], colors[i], label="Target {}".format(i))
        plt.title("Distance Target Training")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 6)
        for i in range(3):
            plt.plot(epochs, vagent[i], colors[i], label="Target {}".format(i))
        plt.title("Distance Target Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.savefig(path + "/results.png")
        plt.show()
        return

    def plot_dists(self, path):
        train_dists = json.load(open(path + "/train-dists.json", "r"))
        val_dists = json.load(open(path + "/val-dists.json", "r"))

        epochs = np.asarray(list(train_dists.keys())).astype(int)
        agent = np.zeros((3, len(epochs)))
        vagent = np.zeros((3, len(epochs)))
        tdists = np.asarray(list(train_dists.values())).astype(float)
        vdists = np.asarray(list(val_dists.values())).astype(float)
        for i in range(3):
            agent[i] = tdists[:, i]
            vagent[i] = vdists[:, i]

        colors = ["k-", "c-", "b-", "g-", "r-", "m-", "y-", "k-"]
        plt.subplot(2, 3, 1)
        plt.plot(epochs, agent[0], "g-", label="Target 0 Distance")
        plt.title("Training")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 2)
        plt.plot(epochs, agent[1], "g-", label="Target 1 Distance")
        plt.title("Training")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 3)
        plt.plot(epochs, agent[2], "g-", label="Target 2 Distance")
        plt.title("Training")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        #####Val
        plt.subplot(2, 3, 4)
        plt.plot(epochs, vagent[0], "m-", label="Target 0 Distance")
        plt.title("Val")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 5)
        plt.plot(epochs, vagent[1], "m-", label="Target 1 Distance")
        plt.title("Val")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 6)
        plt.plot(epochs, vagent[2], "m-", label="Target 2 Distance")
        plt.title("Val")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.savefig(path + "/dists.png")
        plt.show()
        return

########################################################################################################################
# Experimenting with model

if __name__ == '__main__':
    logs = BaselineLogs()

    path = "/Users/dianaescoboza/Documents/PycharmProjects/rl-landmark/rl-medical/src/runs/Aug08_23-45-21_MacBook-Pro.local"
    logs.saveBaseline(path)
    logs.plot_dists(path)

