import torch
import torch.nn as nn
import numpy as np
from .model import BaselineModel
from .data_loader import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

SMOOTH = 1e-6


class DetecTrainer(object):
    def __init__(self, arguments, label_ids, batch_size=64):
        self.batch_size = batch_size
        self.labels = label_ids
        self.max_epochs = arguments.max_episodes
        self.sample = DataLoader(arguments.files, arguments.batch_size).sample()
        self.val_data = DataLoader(arguments.val_files).sample()
        self.LossFunc = BaselineLoss()
        self.data_logger = SummaryWriter()

        # Model
        self.model = BaselineModel()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.train(True)
        if torch.cuda.device_count() > 1:
            print("{} GPUs Available for Training".format(torch.cuda.device_count()))
            self.q_network = nn.DataParallel(self.model)
        self.model.to(self.device)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=arguments.lr)

        return

    def train(self):
        epoch = 0
        for i in range(self.max_epochs):
            self.model.train(True)
            box_losses = []
            loss = 0
            for images, boxes, targets in next(self.sample):
                images = images.to(self.device)
                self.optimizer.zero_grad()
                loc_pred, class_pred = self.model.forward(images.float())
                # Calculate loss
                batch_loss = self.LossFunc(loc_pred, class_pred, boxes, targets)
                batch_loss.backward()

                self.optimizer.step()
                epoch += 1

            self.validation()

        return

    def validation(self):
        self.model.train(False)

        return


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, boxpred, classpred, boxes, labels):
        # Boxes: (x, y, height, width)
        # Class Loss
        lossfunc = nn.BCEWithLogitsLoss()
        _, pred = torch.max(labels, dim=1)
        classloss = lossfunc(classpred, pred)

        # IoU Loss
        # intersection = (boxpred[:2] + boxes[:2])/2
        top_left_t = (boxes[0] - boxes[3]/2, boxes[1] - boxes[2]/2)
        top_left_p = (boxpred[0] - boxpred[3]/2, boxpred[1] - boxpred[2]/2)
        bot_right_t = (boxes[0] + boxes[3]/2, boxes[1] + boxes[2]/2)
        bot_right_p = (boxpred[0] + boxpred[3]/2, boxpred[1] + boxpred[2]/2)
        x1 = max(top_left_t[0], top_left_p[0])
        y1 = max(top_left_t[1], top_left_p[1])
        x2 = max(bot_right_t[0], bot_right_p[0])
        y2 = max(bot_right_t[1], bot_right_p[1])
        interArea = max(0, abs(x1-x2)) * max(0, abs(y1-y2))
        areaBoxes = boxes[2] * boxes[3]
        areaPred = boxpred[2] * boxpred[3]
        iouloss = interArea / (areaBoxes+areaPred-interArea)

        #Location Loss
        locloss = torch.norm(boxes[:2] - boxpred[:2])

        return Variable(classloss + iouloss + locloss).requires_grad(True)


class Args(object):
    def __init__(self):
        self.max_episodes = 10
        self.files = None
        self.lr = 0.001
        self.batch_size = 15


if __name__ == '__main__':
    args = Args()
    agents = [0, 1, 2, 3, 4, 5, 6, 7]
    trainer = DetecTrainer(args, agents)
