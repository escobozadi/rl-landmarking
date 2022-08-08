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
    def __init__(self, arguments, label_ids, setting="baseline", useparallel=True):
        # Hyperparameters
        self.batch_size = arguments.batch_size  # must be multiple of 2 for semi-supervised
        self.labels = label_ids
        self.max_epochs = arguments.max_episodes
        self.semi = False

        # Data Samplers
        self.traindata = DataLoader(arguments.files, landmarks=len(label_ids),
                                    batch_size=arguments.batch_size, learning=setting)
        self.valdata = DataLoader(arguments.val_files, learning=setting)
        self.sample = self.traindata.sample()
        self.val_sample = self.valdata.sample()
        if setting == "semi":
            # For semi-supervised learning
            self.semi = True

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
        self.scaler = torch.cuda.amp.GradScaler()

        # Logger
        self.data_logger = SummaryWriter(comment=arguments.log_comment)
        self.logs = open(self.data_logger.get_logdir() + "/logs.txt", "w")
        print("Using :", self.device)
        self.logs.write("Using :" + str(self.device) + '\n')
        # # self.data_logger.add_hparams({"lr": arguments.lr, "batch_size": arguments.batch_size})
        # _, _, imgs = next(self.sample)
        # imgs = imgs.to(self.device)
        # grid = torchvision.utils.make_grid(imgs)
        # self.data_logger.add_image("images", grid)
        # self.data_logger.add_graph(self.model, input_to_model=imgs)
        # del imgs
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
            self.optimizer.zero_grad(set_to_none=True)
            for n, (targets, boxes, images) in enumerate(self.sample):
                images = images.to(self.device)

                # Model Predictions and Loss Calculation
                loc_pred, class_pred = self.model.forward(images.float())
                loc_pred, class_pred = loc_pred.cpu(), class_pred.cpu()
                if self.semi:
                    # Loss for semi-supervised learning
                    batch_loss = self.LossFunc.unlabelled_loss(loc_pred, class_pred, boxes, targets)
                else:
                    batch_loss = self.LossFunc.boxing_loss(loc_pred, class_pred, boxes, targets)
                batch_loss.backward(torch.ones_like(batch_loss))
                # Optimizer Step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

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
                self.logs.write("Time Taken For {} Epoch: {}".format(epoch, timedelta(seconds=end-start)) + '\n')
                torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/baseline_model.pt")

            # Epoch Results
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for k in dicDist.keys():
                    dicDist[k] = np.nanmean(dicDist[k])
            dis_tar_loss = dis_tar_loss.tolist()

            # Log Epoch Outputs
            self.save_logs(dis_tar_loss, loss, dicDist, epoch)

            # Validation Epoch
            self.validation(epoch)
            del dis_tar_loss, dicDist

        # Save Final Model
        torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/baseline_model.pt")
        self.data_logger.close()
        return

    # @torch.no_grad()
    def validation(self, epoch):
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
            loss_batch = self.LossFunc.boxing_loss(loc_pred, class_pred, boxes, targets)
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
        self.save_logs(dis_tar_loss, loss, dicDist, epoch, "Validation")

        # Save Model With Best Validation Performance
        if loss <= self.best_val:
            print("---Validation Improved!--- \n")
            self.logs.write("Validation Improved!" + '\n')
            torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/best_model.pt")
            self.best_val = loss

        del dicDist, dist
        return

    def save_logs(self, losses_dic, total_loss, agent_dist, epoch, task="Train"):

        self.data_logger.add_scalar(task + " Total Loss", total_loss, epoch)
        self.data_logger.add_scalar(task + " Distance Loss", losses_dic[0], epoch)
        self.data_logger.add_scalar(task + " IoU Loss", losses_dic[1], epoch)
        self.data_logger.add_scalar(task + " Class Loss", losses_dic[2], epoch)
        self.data_logger.add_scalars(task + " Avg Distance", agent_dist, epoch)

        print("EPOCH ", epoch)
        self.logs.write("EPOCH " + str(epoch) + '\n')
        print(task + " Total Loss: {}".format(round(total_loss, 6)))
        self.logs.write(task + " Total Loss: {}".format(round(total_loss, 6)) + '\n')
        print("Target Distance Loss: {}, Box IoU Loss: {}, Class Loss: {}".format(
            round(losses_dic[0], 6), round(losses_dic[1], 6), round(losses_dic[2], 6)))
        self.logs.write("Target Distance Loss: {}, Box IoU Loss: {}, Class Loss: {}".format(
            round(losses_dic[0], 6), round(losses_dic[1], 6), round(losses_dic[2], 6)) + '\n')
        print(task + " AvgDistances:")
        self.logs.write(task + " AvgDistances:" + '\n')
        print("".join(["Agent " + str(k) + ": " + str(round(agent_dist[k], 6)) + " || "
                       for k in agent_dist.keys()]), sep=" ")
        self.logs.write("".join(["Agent " + str(k) + ": " + str(round(agent_dist[k], 6)) + " || "
                                 for k in agent_dist.keys()]) + "  " + '\n')
        return


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.disloss = nn.L1Loss()
        self.lossfunc = nn.BCEWithLogitsLoss(reduce=False)
        self.mse = nn.MSELoss(reduce=False)

    def boxing_loss(self, boxpred, classpred, boxes, labels):
        idx = (labels == 1)
        # boxes = torch.nan_to_num(boxes)
        # Boxes: (x, y, height, width)
        # Class Loss
        pred = ((classpred > 0.5) * 1).double()
        classloss = self.lossfunc(labels.double(), pred)
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
        locloss = self.disloss(torch.nan_to_num(boxes[:, :, :2]), boxpred[:, :, :2])
        locloss = torch.mean(locloss * idx, 1)

        batch_loss = torch.transpose(torch.stack((locloss, iouloss, classloss)), 0, 1)
        return Variable(batch_loss, requires_grad=True)

    def unlabelled_loss(self, boxpred, classpred, boxes, labels):
        """
        labelled image loss = (distance to target loss, IoU loss, class loss)
        unlabelled image loss = labelled data as target
        """
        idxes = [[labels[n, 0].item(), n] for n in range(len(labels))
                 if torch.all(labels[n] == labels[n, 0])]
        labelidx = np.asarray(idxes)[:, 0]
        unlabelidx = np.asarray(idxes)[:, 1]

        # label loss
        label_loss = self.boxing_loss(boxpred[labelidx], classpred[labelidx],
                                      boxes[labelidx], labels[labelidx])
        # Unlabel loss with labeled data as target
        unlabel_loss = self.boxing_loss(boxpred[unlabelidx], classpred[unlabelidx],
                                        boxpred[labelidx], classpred[labelidx])

        batchloss = torch.mean(torch.cat((label_loss, unlabel_loss)), 0)
        return batchloss


class Evaluate(object):
    def __init__(self, arguments, label_ids):
        self.data = DataLoader(arguments.files, landmarks=len(label_ids))


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
        self.log_comment = "tryout"


if __name__ == '__main__':
    args = Args()
    agents = [0, 1, 2, 3, 4, 5, 6, 7]
    trainer = DetecTrainer(args, agents)
    trainer.train()

