import cv2
import numpy as np

class DataLoader(object):
    def __init__(self, files_list, returnLandmarks=True):
        assert files_list, 'There is no file given'
        # read image filenames
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        self.agents = agents
        if self.returnLandmarks:
            self.landmark_files = [
                line.split('\n')[0] for line in open(
                    files_list[1].name)]
            assert len(
                self.image_files) == len(
                self.landmark_files), """number of image files is not equal to
                    number of landmark files"""

    @property
    def num_files(self):
        return len(self.image_files)

    def sample(self, landmark_ids, shuffle=True):
        """ return a random sampled ImageRecord from the list of files
        to sample a new image, should return: image, target loc, file path, spacing
        spacing: consistent unit, mm
        image:  image.dims
        landmark_ids: agent landmark, 0 0 0"""
        if shuffle:
            indexes = np.random.choice(np.arange(self.num_files), self.num_files, replace=False)
            pass
        else:
            indexes = np.arange(self.num_files)
        while True:
            for idx in indexes:
                # png for 2d images np.isnan(m[1]).any()
                sitk_image, image = PngImage().decode(self.image_files[idx])
                if self.returnLandmarks:
                    # transform landmarks to image space if they are in physical space
                    landmark_file = self.landmark_files[idx]
                    landmark = getLandmarksFromTXTFile(landmark_file, self.agents)
                    if np.isnan(landmark[landmark_ids]).all():
                        continue
                    # scaling coor to the size of the image
                    landmark[:, 0] *= image.dims[0]
                    landmark[:, 1] *= image.dims[1]
                    landmarks_round = [np.round(landmark[landmark_ids[i] % 15]) for i in range(self.agents)]
                else:
                    landmarks_round = None

                # extract filename from path, remove .png extension
                image_filenames = [self.image_files[idx][:-4]] * self.agents
                images = [image] * self.agents
                yield (images, landmarks_round, image_filenames)

