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
        self.labels = label_ids  # Landmark ids [0, 1, 2, 3]
        self.max_epochs = arguments.max_episodes
        self.write = arguments.write
        self.semi = False   # Semi or supervised learning
        if setting == "semi":
            # For semi-supervised learning
            self.semi = True

        # Data Samplers
        self.traindata = DataLoader(arguments.files, landmarks=len(label_ids),
                                    batch_size=arguments.batch_size, learning=setting)
        self.valdata = DataLoader(arguments.val_files)
        self.sample = self.traindata.sample()
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
        # if self.write:
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

            # Restart Data Generator
            self.traindata.restartfiles()
            self.sample = self.traindata.sample()

            # Save distance and losses
            dicDist = {}
            for label in self.labels:
                dicDist[str(label)] = 0
            epoch_loss = torch.zeros((len(self.labels), ))
            total_loss = 0

            self.optimizer.zero_grad(set_to_none=True)
            for n, (_, boxes, images) in enumerate(self.sample):
                images = images.to(self.device)

                # Model Predictions and Loss Calculation
                loc_pred = self.model.forward(images.float()).cpu()
                if self.semi:  # Loss for semi-supervised learning
                    dist_loss = \
                        self.LossFunc.unlabelled_loss(loc_pred, boxes)
                else:
                    dist_loss = \
                        self.LossFunc.label_loss(loc_pred, boxes)

                # Backward pass
                dist_loss.backward(torch.ones_like(dist_loss))
                epoch_loss += torch.mean(dist_loss, 0).detach()

                # Optimizer Step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Results for saving
                total_loss += torch.sum(dist_loss).detach()
                for k in dicDist.keys():
                    dist = ((loc_pred[:, int(k), 0] - boxes[:, int(k), 0]) ** 2 +
                            (loc_pred[:, int(k), 1] - boxes[:, int(k), 1]) ** 2).detach().numpy()
                    dicDist[k] += np.nansum(dist)

                # Free up memory
                del images, loc_pred

            epoch += 1
            end = time.time()
            if (epoch == 1) or (epoch % 50 == 0):
                print("Time Taken For {} Epoch: {}".format(epoch, timedelta(seconds=end-start)))
                self.logs.write("Time Taken For {} Epoch: {}".format(epoch, timedelta(seconds=end-start)) + '\n')
                torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/baseline_model.pt")

            # Epoch Results
            total_loss = total_loss / self.traindata.num_files
            epoch_loss = epoch_loss / self.traindata.num_files
            for k in dicDist.keys():
                dicDist[k] = dicDist[k] / self.traindata.num_files

            # Log Epoch Outputs
            self.save_logs(total_loss, dicDist, epoch, epoch_loss.numpy())

            # Validation Epoch
            self.validation(epoch)
            del dicDist

        # Save Final Model
        torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/baseline_model.pt")
        self.data_logger.close()
        return

    def validation(self, epoch):
        self.model.train(False)

        # Restart Data Generator
        self.valdata.restartfiles()
        self.val_sample = self.valdata.sample()

        dicDist = {}
        for label in self.labels:
            dicDist[str(label)] = np.array([])

        val_loss = torch.zeros((len(self.labels),))
        val_total_loss = 0
        for _, boxes, imgs in self.val_sample:
            imgs = imgs.to(self.device)

            # Model Predictions
            loc_pred = self.model.forward(imgs.float()).cpu()

            # Loss Calculation
            dist_loss = self.LossFunc.label_loss(loc_pred, boxes)

            # Loss saving
            val_loss += torch.mean(dist_loss, 0).detach()
            for k in dicDist.keys():
                dist = ((loc_pred[:, int(k), 0] - boxes[:, int(k), 0]) ** 2 +
                        (loc_pred[:, int(k), 1] - boxes[:, int(k), 1]) ** 2).detach().numpy()
                dicDist[k] += np.nansum(dist)

            # Clear Memory Space
            del imgs, loc_pred

        # Validation Epoch Results
        val_total_loss = val_total_loss / self.valdata.num_files
        val_loss = val_loss / self.valdata.num_files
        for k in dicDist.keys():
            dicDist[k] = dicDist[k] / self.valdata.num_files

        # Log Validation Results
        self.save_logs(val_total_loss, dicDist, epoch, val_loss.numpy(), "Validation")

        # Save Model With Best Validation Performance
        if val_total_loss <= self.best_val:
            print("---Validation Improved!--- \n")
            self.logs.write("Validation Improved!" + '\n')
            torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/best_model.pt")
            self.best_val = val_total_loss

        del dicDist, dist
        return

    def save_logs(self, total_loss, agent_dist, epoch, landarks_losses, task="Train"):

        # self.data_logger.add_scalar(task + " Total Loss", total_loss, epoch)
        # self.data_logger.add_scalars(task + " Avg Distance", agent_dist, epoch)

        self.logs.write("EPOCH " + str(epoch) + '\n')
        self.logs.write(task + " Total Loss: {}".format(total_loss) + '\n')
        # self.logs.write("Target Distance Loss: {}, Class Loss: {}".format(
        #     losses_dic[0], losses_dic[1]) + '\n')
        self.logs.write("".join(["Agent {} Loss: ".format(k) + str(landarks_losses[k]) + " || " + " \n"
                                 for k in range(len(self.labels))]) + '\n')
        self.logs.write(task + " AvgDistances:" + '\n')
        self.logs.write("".join(["Agent " + str(k) + ": " + str(agent_dist[k]) + " || "
                                 for k in agent_dist.keys()]) + "  " + "\n")

        print("EPOCH ", epoch)
        print(task + " Total Loss: {}".format(total_loss))
        # print("Target Distance Loss: {}, Class Loss: {}".format(
        #     losses_dic[0], losses_dic[1]))
        print("".join(["Agent {} Dist Loss: ".format(k) + str(landarks_losses[k]) + " || " + " \n"
                       for k in range(len(landarks_losses))]))
        print(task + " AvgDistances:")
        print("".join(["Agent " + str(k) + ": " + str(agent_dist[k]) + " || "
                       for k in agent_dist.keys()]), sep=" ")

        return


class BaselineLoss(nn.Module):
    def __init__(self, landmarks=3):
        super(BaselineLoss, self).__init__()
        self.disloss = nn.L1Loss()
        self.classlossfunc = nn.BCEWithLogitsLoss(reduce=False, reduction='none')
        self.mse = nn.MSELoss()
        self.targets = landmarks

    @staticmethod
    def IoU(boxpred, boxes):
        # IoU Loss
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
        return iouloss

    def clssLoss(self, classpred, labels):
        # Class Loss
        landmarks_pred = ((classpred > 0.5) * 1).detach().double()
        classloss = torch.mean(self.classlossfunc(classpred, labels.double()), 0)
        classloss = Variable(classloss, requires_grad=True)

        return

    def label_loss(self, coorpred, coor, labels=None):
        # Location Loss
        # loss = []
        # dists = []
        coor = torch.nan_to_num(coor)
        dists = torch.norm(coorpred-coor, dim=2)
        # for s in range(self.targets):
        #     loss.append(self.mse(coor[:, s], coorpred[:, s]))
        #     locloss = ((coor[:, s, 0]) - (coorpred[:, s, 0])) ** 2 + \
        #               ((coor[:, s, 1]) - (coorpred[:, s, 1])) ** 2
        #     dists.append(torch.mean(locloss))
        # totloss = torch.tensor(loss) + torch.tensor(dists)
        # return Variable(torch.tensor(dists), requires_grad=True)
        return dists

    def unlabelled_loss(self, boxpred, boxes, labels=None):
        """
        labelled image loss = (distance to target loss, IoU loss, class loss)
        unlabelled image loss = labelled data as target
        """
        idxes = [[labels[n, 0].item(), n] for n in range(len(labels))
                 if torch.all(labels[n] == labels[n, 0])]
        labelidx = np.asarray(idxes)[:, 0]
        unlabelidx = np.asarray(idxes)[:, 1]

        # label loss
        label_loss = self.boxing_loss(boxpred[labelidx],
                                      boxes[labelidx], labels[labelidx])
        # Unlabel loss with labeled data as target
        unlabel_loss = self.boxing_loss(boxpred[unlabelidx],boxpred[labelidx])

        loss = torch.cat((label_loss[0], unlabel_loss[0]), 0)
        classloss = torch.cat((label_loss[1], unlabel_loss[1]), 0)
        label_preds = torch.cat((label_loss[2], unlabel_loss[2]), 0)
        return loss, classloss, label_preds


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
        self.write = False


if __name__ == '__main__':
    args = Args()
    # agents = [0, 1, 2, 3, 4, 5, 6, 7]
    # trainer = DetecTrainer(args, agents)
    # trainer.train()
    """
    train total loss 1, 40, 80
        0.024934755638241768
        0.024884937331080437
        0.02496100589632988
        0.024960510432720184
        
    val total loss 1, 40, 80
        0.681381344795227
        0.6813812851905823
        0.681381344795227 
        0.6813812851905823 
    
    train epoch 1, 40, 80
    target 0: 0.4451649094918906
              0.4451649094918906
    target 1: 0.7605255155369293
              0.7605255155369293
    target 2: 0.32714605707306044
              0.32714605707306044
    
    val epoch 1, 40, 80
    target 0: 0.428228833927558
              0.428228833927558
    target 1: 0.7641626072044556
              0.7641626072044556
    target 2: 0.338140210211277
              0.338140210211277
    """


