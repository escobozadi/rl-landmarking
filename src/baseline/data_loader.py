import cv2
import numpy as np


class DataLoader(object):
    def __init__(self, files_list, batch_size=1, returnLandmarks=True):
        assert files_list, 'There is no files given'
        # read image filenames
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
        self.batch_size = batch_size
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        if self.returnLandmarks:
            self.landmark_files = [
                line.split('\n')[0] for line in open(
                    files_list[1].name)]
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
        landmarks = np.zeros((8, 4))
        landmarks[:] = np.nan

        with open(file) as t:
            lines = [x.strip() for x in list(t) if x]
            for l in lines:
                info = l.split(" ")
                id = int(info[0])
                landmarks[id, :] = info[1:]

        landmarks = np.asarray(landmarks)  # (# labels, x, y, height, width)
        targets = np.argwhere(~np.isnan(landmarks[:, 0])).reshape([-1, ])
        return landmarks, targets

    def decode(self, filename):
        ''' return image object with info'''
        image = ImageRecord()
        image.name = filename

        np_image = cv2.imread(filename)
        if np_image is None:
            print("Empty Image")
            print(image.name)
        image.dims = np_image.shape  # (x,y)
        np_image = cv2.copyMakeBorder(np_image, 0, 786 - np_image.shape[0], 0, 1136 - np_image.shape[1], cv2.BORDER_CONSTANT)
        np_image = np_image.transpose(2, 0, 1)  # (channels, x, y)
        image.data = np_image

        return image

    @property
    def num_files(self):
        return len(self.image_files)

    def sample(self, landmark_ids=None, shuffle=True):
        if shuffle:
            indexes = np.random.choice(np.arange(self.num_files), self.batch_size, replace=False)
        else:
            indexes = np.arange(self.batch_size)
        images = []
        landmarks = []
        targets = []
        while True:
            for idx in indexes:
                image = self.decode(self.image_files[idx])
                images.append(image)
                if self.returnLandmarks:
                    landmark, target = self.getLandmarksFromTXTFile(self.landmark_files[idx])  # 8x4, #, 1
                    # if np.isnan(landmarks[landmark_ids]).all():
                    #     continue
                    # scaling coor to the size of the image
                    # landmark[:, 0] *= image.dims[0]
                    # landmark[:, 1] *= image.dims[1]
                    # landmarks_round = [np.round(landmark[landmark_ids[i] % 15])for i in range(self.agents)]
                else:
                    landmark = None
                    target = None
                landmarks.append(landmark)
                targets.append(target)
                # extract filename from path, remove .png extension
                # image_filenames = [self.image_files[idx][:-4]] * self.agents
                # images = [image] * self.agents
            yield images, landmarks, targets


class ImageRecord(object):
    pass

