import copy

import cv2
import numpy as np
import torch


class DataLoader(object):
    def __init__(self, files_list, landmarks=8, batch_size=2, learning="base", returnLandmarks=True):
        assert files_list, 'There is no files given'

        self.batch_size = batch_size
        self.semi = False
        if learning == "semi":
            self.batch_size = int(self.batch_size / 2)
            self.semi = True
        self.landmarks = landmarks
        self.returnLandmarks = returnLandmarks
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0]
                                   for line in open(files_list[1].name)]

        self.files_idxes = np.arange(self.num_files)
        self.batchidx = None
        self.restartfiles()
        assert len(self.image_files) == len(self.landmark_files), """number of image files is not equal to
                    number of landmark files"""

    def getLandmarks(self, file, split=' '):
        """
        0  femur
        1  patella
        2  tendon
        3  tibia/fibula
        4  talus
        5  ulna
        6  triceps tendon insertion
        7  humerus
        """
        # (# labels, x, y, height, width)
        landmarks = np.zeros((self.landmarks, 4))
        landmarks[:] = np.nan

        classes = [0 for i in range(self.landmarks)]
        with open(file) as t:
            lines = [x.strip() for x in list(t) if x]
            for l in lines:
                info = l.split(" ")
                id = int(info[0])
                landmarks[id, :] = info[1:]

        targets = np.argwhere(~np.isnan(landmarks[:, 0])).reshape([-1, ])
        for i in range(self.landmarks):
            if i in targets:
                classes[i] = 1
        return landmarks.tolist(), classes

    def getImage(self, filename):
        # Read Image and Get to Right Size
        np_image = cv2.imread(filename)
        if np_image is None:
            print("Empty Image")
            print(filename)
        # # Max size image = 786 x 1136 x 3
        # np_image = cv2.copyMakeBorder(np_image, 0, 786 - np_image.shape[0],
        #                               0, 1136 - np_image.shape[1], cv2.BORDER_CONSTANT)
        np_image = np_image / 255.0
        np_image = np_image.transpose(2, 0, 1)  # (channels, x, y)
        return np_image.tolist()

    def getNoisyImage(self, image):
        noisy_image = np.asarray(copy.deepcopy(image))
        size = noisy_image.shape
        noisy_image = noisy_image + np.random.rand(size[0], size[1], size[2]) * 0.5
        return noisy_image.tolist()

    @property
    def num_files(self):
        return len(self.image_files)

    def restartfiles(self, shuffle=True):
        self.files_idxes = np.arange(self.num_files)
        if shuffle:
            np.random.shuffle(self.files_idxes)
        self.batchidx = []
        while len(self.files_idxes) != 0:
            batch = self.files_idxes[:min(len(self.files_idxes), self.batch_size)]
            self.files_idxes = self.files_idxes[len(batch):]
            self.batchidx.append(batch)
        return

    def mixSample(self, images, landmarks, classes):
        unlabelidx = [(2*i)+1 for i in range(int(len(classes)/2))]
        labelidx = [(2*i) for i in range(int(len(classes)/2))]

        # shuffle data idx
        idxes = np.arange(len(classes))
        np.random.shuffle(idxes)
        newunlabel = [np.where(idxes == n)[0].item() for n in unlabelidx]
        newlabel = [np.where(idxes == n)[0].item() for n in labelidx]
        # shuffle batch
        images = np.asarray(images)[idxes]
        landmarks = np.asarray(landmarks)[idxes]
        classes = np.asarray(classes)[idxes]

        for i in range(len(newunlabel)):
            # keep track of the image the noisy image comes from
            classes[newunlabel[i]] = newlabel[i]

        return images.tolist(), landmarks.tolist(), classes.tolist()

    def sample(self):
        for batch in self.batchidx:
            images = []
            landmarks = []
            targets = []
            for i in batch:
                imagefile = self.image_files[i]
                landmarkfile = self.landmark_files[i]
                image = self.getImage(imagefile)
                landmark, target = self.getLandmarks(landmarkfile)
                images.append(image)
                landmarks.append(landmark)
                targets.append(target)

                # Noisy Images for Semi Supervised
                if self.semi:
                    noisy_img = self.getNoisyImage(image)
                    noisy_land = np.zeros((self.landmarks, 4)).tolist()
                    noisy_targ = [0 for i in range(self.landmarks)]
                    images.append(noisy_img)
                    landmarks.append(noisy_land)
                    targets.append(noisy_targ)
            if self.semi:
                images, landmarks, targets = self.mixSample(images, landmarks, targets)
            targets = torch.as_tensor(targets)
            landmarks = torch.as_tensor(landmarks)
            images = torch.as_tensor(np.asarray(images))
            yield targets, landmarks, images


