import torch
import torch.nn as nn
from baseline import BaselineModel
from data_loader import DataLoader
SMOOTH = 1e-6

class DetecTrainer(object):
    def __init__(self, args, label_ids, batch_size=64):
        self.batch_size = batch_size
        self.labels = label_ids
        self.model = BaselineModel()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_epochs = args.max_episodes
        self.model.train(True)
        self.data = DataLoader(args.landmarks)

        return

    def train(self):
        epoch = 0
        for i in range(self.max_epochs):
            box_losses = []
            loss = 0
            for images, boxes, ids in next(self.data.sample()):
                images = images.to(self.device)



        return

    def validation(self):
        return

    def iou_loss(outputs: torch.Tensor, labels: torch.Tensor):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        out = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (out & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (out | labels).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded  # Or thresholded.mean() if you are interested in average across the batch

    def one_hot_ce_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        _, labels = torch.max(targets, dim=1)
        return criterion(outputs, labels)

