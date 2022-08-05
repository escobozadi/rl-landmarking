import torch
import time
import torch.nn as nn
import numpy as np
import warnings
import torchvision.utils
from datetime import timedelta

from .model import BaselineModel
from .data_loader import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class DetecTrainer(object):
    def __init__(self, arguments, label_ids, useparallel=False):
        # Hyperparameters
        self.batch_size = arguments.batch_size
        self.labels = label_ids
        self.max_epochs = arguments.max_episodes

        # Data Samplers
        self.traindata = DataLoader(arguments.files, landmarks=len(label_ids),
                                    batch_size=arguments.batch_size)
        self.sample = self.traindata.sample()
        self.valdata = DataLoader(arguments.val_files)
        self.val_sample = self.valdata.sample()

        # Model
        self.model = BaselineModel()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.train(True)
        if useparallel:
            if torch.cuda.device_count() > 1:
                print("{} GPUs Available for Training".format(torch.cuda.device_count()))
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # Loss Function and Optimizer
        self.LossFunc = BaselineLoss()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=arguments.lr)
        self.best_val = float('inf')

        # Logger
        self.data_logger = SummaryWriter(comment="BaselineModel")
        # self.data_logger.add_hparams({"lr": arguments.lr, "batch_size": arguments.batch_size})
        _, _, imgs = next(self.sample)
        imgs = imgs.to(self.device)
        grid = torchvision.utils.make_grid(imgs)
        self.data_logger.add_image("images", grid)
        self.data_logger.add_graph(self.model, input_to_model=imgs)
        del imgs
        # self.data_logger.close()

        return

    def train(self):

        epoch = 0
        start = time.time()
        for i in range(self.max_epochs):
            self.model.train(True)
            # Restart Generator
            self.traindata.restartfiles()
            self.sample = self.traindata.sample()
            dicDist = {}
            for label in self.labels:
                dicDist[str(label)] = np.array([])

            loss = 0
            dis_tar_loss = torch.zeros((3,))
            for targets, boxes, images in self.sample:
                images = images.to(self.device)
                self.optimizer.zero_grad()

                # Model Predictions and Loss Calculation
                loc_pred, class_pred = self.model.forward(images.float())
                loc_pred, class_pred = loc_pred.cpu(), class_pred.cpu()
                batch_loss = self.LossFunc(loc_pred, class_pred, boxes, targets)
                batch_loss.backward(torch.ones_like(batch_loss))
                self.optimizer.step()

                # Results saving
                loss += torch.mean(batch_loss).detach().item()
                dis_tar_loss += torch.mean(batch_loss, 0).detach()
                dist = torch.norm(loc_pred[:, :, :2] - boxes[:, :, :2],
                                  dim=2).detach().numpy()
                for k in dicDist.keys():
                    dicDist[k] = np.append(dicDist[k], dist[:, int(k)])

                # Free up memory
                del images, loc_pred, class_pred, batch_loss

            epoch += 1
            end = time.time()
            if (epoch == 1) or (epoch % 100 == 0):
                print("Time Taken For {} Epoch: {}".format(epoch, timedelta(seconds=end-start)))

            # Epoch Results
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for k in dicDist.keys():
                    dicDist[k] = np.nanmean(dicDist[k])
            dis_tar_loss = dis_tar_loss.tolist()

            # Print Results to Terminal
            print("EPOCH ", epoch)
            print("Training Total Loss: {}".format(round(loss, 6)))
            print("Target Distance Loss: {}, Box IoU Loss: {}, Class Loss: {}".format(
                round(dis_tar_loss[0], 6), round(dis_tar_loss[1], 6), round(dis_tar_loss[2], 6)))
            print("Training AvgDistances:")
            print("".join(["Agent " + str(k) + ": " + str(round(dicDist[k], 6)) + " || "
                           for k in dicDist.keys()]), sep=" ")

            # Log Epoch Outputs
            self.data_logger.add_scalar("Train Total Loss", loss, epoch)
            self.data_logger.add_scalar("Train Distance Loss", dis_tar_loss[0], epoch)
            self.data_logger.add_scalar("Train IoU Loss", dis_tar_loss[1], epoch)
            self.data_logger.add_scalar("Train Class Loss", dis_tar_loss[2], epoch)
            self.data_logger.add_scalars("Train Avg Distance", dicDist, epoch)

            # Validation Epoch
            self.validation()
            del dis_tar_loss, dicDist

        # Save Final Model
        torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/baseline_model.pt")
        self.data_logger.close()
        return

    def validation(self):
        self.model.train(False)
        # Restart Generator
        self.valdata.restartfiles()
        self.val_sample = self.valdata.sample()

        dicDist = {}
        for label in self.labels:
            dicDist[str(label)] = np.array([])

        loss = 0
        dis_tar_loss = torch.zeros((3,))
        for targets, boxes, imgs in self.val_sample:
            imgs = imgs.to(self.device)

            # Model Predictions and Loss Calculation
            loc_pred, class_pred = self.model.forward(imgs.float())
            loc_pred, class_pred = loc_pred.cpu(), class_pred.cpu()
            loss_batch = self.LossFunc(loc_pred, class_pred, boxes, targets)
            loss += torch.mean(loss_batch).detach().item()

            # Batch Loss: Batch size x (Distance Loss, IoU Loss, Class Loss)
            dis_tar_loss += torch.mean(loss_batch, 0).detach()
            dist = torch.norm(loc_pred[:, :, :2] - boxes[:, :, :2],
                              dim=2).detach().numpy()
            for i in dicDist.keys():
                dicDist[i] = np.append(dicDist[i], dist[:, int(i)])

            # Clear Memory Space
            del imgs, loc_pred, class_pred, loss_batch

        # Validation Epoch Results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k in dicDist.keys():
                dicDist[k] = np.nanmean(dicDist[k])
        dis_tar_loss = dis_tar_loss.tolist()

        # Log Validation Results
        self.data_logger.add_scalar("Val Total Loss", loss)
        self.data_logger.add_scalar("Val Distance Loss", dis_tar_loss[0])
        self.data_logger.add_scalar("Val IoU Loss", dis_tar_loss[1])
        self.data_logger.add_scalar("Val Class Loss", dis_tar_loss[2])
        self.data_logger.add_scalars("Val Avg Distance", dicDist)

        # Save Model With Best Validation Performance
        if loss <= self.best_val:
            print("Validation Improved!")
            torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/best_model.pt")
            self.best_val = loss

        # Print Results to Terminal
        print("Validation Epoch, Total Loss: {}".format(round(loss, 6)))
        print("Target Distance Loss: {}, Box IoU Loss: {}, Class Loss: {}".format(
            round(dis_tar_loss[0], 6), round(dis_tar_loss[1], 6), round(dis_tar_loss[2], 6)))
        print("Validation AvgDistances:")
        print("".join(["Agent " + str(k) + ": " + str(round(dicDist[k], 6)) + " || "
                       for k in dicDist.keys()]), sep=" ")
        del dicDist, dist
        return


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, boxpred, classpred, boxes, labels):
        idx = (labels == 1)
        # Boxes: (x, y, height, width)
        # Class Loss
        lossfunc = nn.BCEWithLogitsLoss(reduce=False)
        pred = ((classpred > 0.5) * 1).double()
        classloss = lossfunc(labels.double(), pred)
        classloss = torch.mean(classloss, 1)

        # IoU Loss
        # intersection = (boxpred[:2] + boxes[:2])/2
        top_left_t = torch.stack((boxes[:, :, 0] - boxes[:, :, 3]/2,
                      boxes[:, :, 1] - boxes[:, :, 2]/2), 2)
        top_left_p = torch.stack((boxpred[:, :, 0] - boxpred[:, :, 3]/2,
                      boxpred[:, :, 1] - boxpred[:, :, 2]/2), 2)
        bot_right_t = torch.stack((boxes[:, :, 0] + boxes[:, :, 3]/2,
                       boxes[:, :, 1] + boxes[:, :, 2]/2), 2)
        bot_right_p = torch.stack((boxpred[:, :, 0] + boxpred[:, :, 3]/2,
                       boxpred[:, :, 1] + boxpred[:, :, 2]/2), 2)
        x1 = torch.max(top_left_t[:, :, 0], top_left_p[:, :, 0])
        y1 = torch.max(top_left_t[:, :, 1], top_left_p[:, :, 1])
        x2 = torch.min(bot_right_t[:, :, 0], bot_right_p[:, :, 0])
        y2 = torch.min(bot_right_t[:, :, 1], bot_right_p[:, :, 1])
        interArea = torch.sub(x1, x2) * torch.sub(y1, y2)
        interArea[(interArea < 0).nonzero(as_tuple=True)] = 0
        areaBoxes = boxes[:, :, 2] * boxes[:, :, 3]
        areaPred = boxpred[:, :, 2] * boxpred[:, :, 3]
        iouloss = interArea / (areaBoxes+areaPred-interArea)
        iouloss = torch.nanmean(iouloss, 1)


        # Location Loss
        disloss = nn.L1Loss()
        # locloss = torch.norm(torch.nan_to_num(boxes[:, :, :2]) - boxpred[:, :, :2],
        #                      dim=2)
        locloss = disloss(torch.nan_to_num(boxes[:, :, :2]), boxpred[:, :, :2])
        locloss = torch.mean(locloss * idx, 1)

        # batch_loss = torch.transpose(torch.stack((iouloss+locloss, classloss)), 0, 1)
        batch_loss = torch.transpose(torch.stack((locloss, iouloss, classloss)), 0, 1)
        return Variable(batch_loss, requires_grad=True)


class Evaluate(object):
    def __init__(self):
        pass


#######################################################################################################################
# Ignore; Experimenting with Code

class Args(object):
    def __init__(self):
        self.max_episodes = 2
        self.files = ["../data/filenames/local_images.txt",
                      "../data/filenames/local_landmarks.txt"]
        self.val_files = ["../data/filenames/local_images.txt",
                          "../data/filenames/local_landmarks.txt"]
        self.lr = 0.001
        self.batch_size = 2


if __name__ == '__main__':
    args = Args()
    agents = [0, 1, 2, 3, 4, 5, 6, 7]
    trainer = DetecTrainer(args, agents)
    trainer.train()

