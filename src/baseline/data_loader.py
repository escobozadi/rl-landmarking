import cv2
import numpy as np


class DataLoader(object):
    def __init__(self, files_list, batch_size=1, returnLandmarks=True):
        assert files_list, 'There is no file given'
        # read image filenames
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
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
        Extract each landmark point line by line from a text file, and return vector containing all landmarks.
        0  femur
        2  tendon
        1  patella
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

        landmarks = np.asarray(landmarks)  # (# labels, x, y, width, height)
        targets = np.argwhere(~np.isnan(landmarks[:, 0])).reshape([-1, ])
        # landmarks[:,0] = -landmarks[:,0]
        # landmarks = landmarks.reshape((-1, landmarks.shape[1]))
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

    def sample(self, landmark_ids=None, batch_size=None, shuffle=True):
        """ return a random sampled ImageRecord from the list of files
        to sample a new image, should return: image, target loc, file path, spacing
        spacing: consistent unit, mm
        image:  image.dims
        landmark_ids: agent landmark, 0 0 0"""
        if shuffle:
            # indexes = np.random.choice(np.arange(self.num_files), batch_size, replace=False)
            indexes = np.random.choice(np.arange(self.num_files), self.num_files, replace=False)
        else:
            indexes = np.arange(self.num_files)
        while True:
            for idx in indexes:
                image = self.decode(self.image_files[idx])
                if self.returnLandmarks:
                    # transform landmarks to image space if they are in physical space
                    landmark_file = self.landmark_files[idx]
                    landmarks, targets = self.getLandmarksFromTXTFile(landmark_file)  # 8x4, #, 1
                    # if np.isnan(landmarks[landmark_ids]).all():
                    #     continue
                    # scaling coor to the size of the image
                    # landmark[:, 0] *= image.dims[0]
                    # landmark[:, 1] *= image.dims[1]
                    # landmarks_round = [np.round(landmark[landmark_ids[i] % 15])for i in range(self.agents)]
                else:
                    landmarks = None
                    targets = None
                # extract filename from path, remove .png extension
                # image_filenames = [self.image_files[idx][:-4]] * self.agents
                # images = [image] * self.agents
                yield image, landmarks, targets


class ImageRecord(object):
    pass

