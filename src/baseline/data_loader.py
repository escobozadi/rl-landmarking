import cv2
import numpy as np
import torch


class DataLoader(object):
    def __init__(self, files_list, landmarks=8, batch_size=1, returnLandmarks=True):
        assert files_list, 'There is no files given'

        self.batch_size = batch_size
        self.landmarks = landmarks
        self.returnLandmarks = returnLandmarks
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0]
                                   for line in open(files_list[1].name)]
        # self.image_files = [line.split('\n')[0]
        #                     for line in open(files_list[0])]
        # self.landmark_files = [line.split('\n')[0]
        #                        for line in open(files_list[1])]

        self.files_idxes = np.arange(len(self.image_files))
        assert len(self.image_files) == len(self.landmark_files), """number of image files is not equal to
                    number of landmark files"""

    def getLandmarksFromTXTFile(self, file, split=' '):
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
        landmarks = np.zeros((self.landmarks, 4))
        landmarks[:] = np.nan
        # classes = np.zeros((self.landmarks, ))
        classes = [0 for i in range(self.landmarks)]
        with open(file) as t:
            lines = [x.strip() for x in list(t) if x]
            for l in lines:
                info = l.split(" ")
                id = int(info[0])
                landmarks[id, :] = info[1:]

        # landmarks = np.asarray(landmarks)  # (# labels, x, y, height, width)
        targets = np.argwhere(~np.isnan(landmarks[:, 0])).reshape([-1, ])
        for i in range(self.landmarks):
            if i in targets:
                classes[i] = 1
        return landmarks.tolist(), classes

    def decode(self, filename):
        np_image = cv2.imread(filename)
        if np_image is None:
            print("Empty Image")
            print(filename)
        np_image = cv2.copyMakeBorder(np_image, 0, 786 - np_image.shape[0],
                                      0, 1136 - np_image.shape[1], cv2.BORDER_CONSTANT)
        np_image = np_image.transpose(2, 0, 1)  # (channels, x, y)
        return np_image

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

    def sample(self):
        for batch in self.batchidx:
            images = []
            landmarks = []
            targets = []
            for i in batch:
                imagefile = self.image_files[i]
                landmarkfile = self.landmark_files[i]
                image = self.decode(imagefile)
                landmark, target = self.getLandmarksFromTXTFile(landmarkfile)
                images.append(image)
                landmarks.append(landmark)
                targets.append(target)
            targets = torch.as_tensor(targets)
            landmarks = torch.as_tensor(landmarks)
            images = torch.as_tensor(np.asarray(images))
            yield targets, landmarks, images

