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
                dicDist[str(label)] = np.array([])
            # loss = 0
            dis_tar_loss = torch.zeros((2,))
            epoch_loss = torch.zeros((len(self.labels), 2))

            self.optimizer.zero_grad(set_to_none=True)
            for n, (targets, boxes, images) in enumerate(self.sample):
                images = images.to(self.device)

                # Model Predictions and Loss Calculation
                loc_pred, class_pred = self.model.forward(images.float())
                loc_pred, class_pred = loc_pred.cpu(), class_pred.cpu()
                if self.semi:
                    # Loss for semi-supervised learning
                    dist_loss, class_loss, landmarks_pred = \
                        self.LossFunc.unlabelled_loss(loc_pred, class_pred, boxes, targets)
                else:
                    dist_loss, class_loss, landmarks_pred = \
                        self.LossFunc.label_loss(loc_pred, class_pred, boxes, targets)

                # Backward pass
                class_loss.backward(torch.ones_like(class_loss))
                for var in dist_loss:
                    var.backward(torch.ones_like(var))

                batch_loss = torch.zeros((len(self.labels), 2))
                batch_loss[:, 0] = torch.as_tensor(dist_loss)
                batch_loss[:, 1] = class_loss.detach()
                epoch_loss += batch_loss

                # Optimizer Step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Results for saving
                # loss += torch.sum(torch.mean(batch_loss, 0)).detach().item()
                dis_tar_loss += torch.sum(batch_loss, 0).detach()
                dist = ((loc_pred[:, :, 0] - boxes[:, :, 0]) ** 2 +
                        (loc_pred[:, :, 1] - boxes[:, :, 1]) ** 2).detach().numpy()
                for k in dicDist.keys():
                    dicDist[k] = np.append(dicDist[k], dist[:, int(k)])

                # Free up memory
                del images, loc_pred, class_pred, batch_loss

            epoch += 1
            end = time.time()
            dis_tar_loss = dis_tar_loss / self.traindata.num_files
            epoch_loss = epoch_loss / self.traindata.num_files
            loss = torch.sum(dis_tar_loss).item()
            if (epoch == 1) or (epoch % 25 == 0):
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
            self.save_logs(dis_tar_loss, loss, dicDist, epoch, epoch_loss.numpy())

            # Validation Epoch
            self.validation(epoch)
            del dis_tar_loss, dicDist

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

        val_loss = torch.zeros((len(self.labels), 2))
        dis_tar_loss = torch.zeros((2,))
        for targets, boxes, imgs in self.val_sample:
            imgs = imgs.to(self.device)

            # Model Predictions
            loc_pred, class_pred = self.model.forward(imgs.float())
            loc_pred, class_pred = loc_pred.cpu(), class_pred.cpu()

            # Loss Calculation
            dist_loss, class_loss, landmarks_pred = self.LossFunc.label_loss(loc_pred, class_pred, boxes, targets)

            # Loss saving
            loss_batch = torch.zeros((len(self.labels), 2))
            loss_batch[:, 0] = torch.as_tensor(dist_loss)
            loss_batch[:, 1] = class_loss.detach()
            val_loss += loss_batch.detach()
            # loss += torch.sum(torch.mean(loss_batch, 0)).detach().item()

            # Batch Loss: Batch size x (Distance Loss, Class Loss)
            dis_tar_loss += torch.sum(loss_batch, 0).detach()
            dist = ((loc_pred[:, :, 0] - boxes[:, :, 0]) ** 2 +
                    (loc_pred[:, :, 1] - boxes[:, :, 1]) ** 2).detach().numpy()
            for i in dicDist.keys():
                dicDist[i] = np.append(dicDist[i], dist[:, int(i)])

            # Clear Memory Space
            del imgs, loc_pred, class_pred, loss_batch

        dis_tar_loss = dis_tar_loss / self.valdata.num_files
        val_loss = val_loss / self.valdata.num_files
        loss = torch.sum(dis_tar_loss).item()

        # Validation Epoch Results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k in dicDist.keys():
                dicDist[k] = np.nanmean(dicDist[k])
        dis_tar_loss = dis_tar_loss.tolist()

        # Log Validation Results
        self.save_logs(dis_tar_loss, loss, dicDist, epoch, val_loss.numpy(), "Validation")

        # Save Model With Best Validation Performance
        if loss <= self.best_val:
            print("---Validation Improved!--- \n")
            self.logs.write("Validation Improved!" + '\n')
            torch.save(self.model.state_dict(), self.data_logger.get_logdir() + "/best_model.pt")
            self.best_val = loss

        del dicDist, dist
        return

    def save_logs(self, losses_dic, total_loss, agent_dist, epoch, landarks_losses, task="Train"):

        self.data_logger.add_scalar(task + " Total Loss", total_loss, epoch)
        self.data_logger.add_scalar(task + " Distance Loss", losses_dic[0], epoch)
        self.data_logger.add_scalar(task + " Class Loss", losses_dic[1], epoch)
        self.data_logger.add_scalars(task + " Avg Distance", agent_dist, epoch)

        self.logs.write("EPOCH " + str(epoch) + '\n')
        self.logs.write(task + " Total Loss: {}".format(total_loss) + '\n')
        self.logs.write("Target Distance Loss: {}, Class Loss: {}".format(
            round(losses_dic[0], 6), round(losses_dic[1], 6)) + '\n')
        self.logs.write("".join(["Agent {} Dist Loss: ".format(k) + str(round(landarks_losses[k, 0], 6)) + " || " +
                                 "Detection Loss: " + str(round(landarks_losses[k, 1], 6)) + " \n"
                                 for k in range(len(self.labels))]) + '\n')
        self.logs.write(task + " AvgDistances:" + '\n')
        self.logs.write("".join(["Agent " + str(k) + ": " + str(round(agent_dist[k], 6)) + " || "
                                 for k in agent_dist.keys()]) + "  " + "\n")

        print("EPOCH ", epoch)
        print(task + " Total Loss: {}".format(total_loss))
        print("Target Distance Loss: {}, Class Loss: {}".format(
            round(losses_dic[0], 6), round(losses_dic[1], 6)))
        print("".join(["Agent {} Dist Loss: ".format(k) + str(round(landarks_losses[k][0], 6)) + " || " +
                       "Detection Loss: " + str(round(landarks_losses[k][1], 6)) + " \n"
                       for k in range(len(landarks_losses))]))
        print(task + " AvgDistances:")
        print("".join(["Agent " + str(k) + ": " + str(round(agent_dist[k], 6)) + " || "
                       for k in agent_dist.keys()]), sep=" ")

        return


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        # self.disloss = nn.L1Loss()
        self.classlossfunc = nn.BCEWithLogitsLoss(reduce=False, reduction='none')
        self.mse = nn.MSELoss(reduce=False)

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

    def label_loss(self, coorpred, classpred, coor, labels):
        idx = (labels == 1)

        # Class Loss
        landmarks_pred = ((classpred > 0.5) * 1).detach().double()
        classloss = torch.mean(self.classlossfunc(classpred, labels.double()), 0)
        classloss = Variable(classloss, requires_grad=True)

        # Location Loss
        loss = []
        coor = torch.nan_to_num(coor)
        for s in range(labels.shape[1]):
            locloss = (coor[:, s, 0] - coorpred[:, s, 0]) ** 2 + \
                      (coor[:, s, 1] - coorpred[:, s, 1]) ** 2
            loss.append(Variable(torch.mean(locloss), requires_grad=True))
        return loss, classloss, landmarks_pred

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
    agents = [0, 1, 2, 3, 4, 5, 6, 7]
    trainer = DetecTrainer(args, agents)
    trainer.train()

