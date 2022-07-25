import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import copy


class CleanData(object):
    def __init__(self, dir, landmarks_dir):
        self.images_dir = dir
        self.landmarks_dir = landmarks_dir
        self.images_files = [f for f in os.listdir(dir) if not f.startswith('.')]
        self.landmark_files = [f for f in os.listdir(landmarks_dir) if not f.startswith('.')]

    def ImageNorm(self, path, destination):

        idx = 0
        for im in self.images_files:
            image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Image is empty: {}, {}".format(idx, im))
            im_norm = np.zeros(image.shape)
            im_norm = cv2.normalize(image, im_norm, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(destination + im, im_norm)
            idx += 1

        return

    def RescaleImages(self):

        scale_percent = 60  # percent of original size
        for im in self.images_files:
            img = cv2.imread(self.images_dir + im, cv2.IMREAD_UNCHANGED)
            # print('Original Dimensions : ', img.shape)
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            # print('Resized Dimensions : ', resized.shape)
            cv2.imwrite(self.images_dir + im, resized)

        return

    def ModelFilenames(self):
        idx = np.arange(len(self.images_files))
        portion = 0.1  # 80/10/10 -> train/val/test

        val = np.random.choice(idx, size=int(len(idx)*portion))
        val = np.sort(val)
        tst = np.random.choice(np.delete(idx, val), size=int(len(idx) * portion))
        tst = np.sort(tst)
        idx = np.delete(idx, np.append(tst, val))

        # Image files
        with open("src/data/filenames/images_test.txt", "w") as f:
            for file in np.asarray(self.images_files)[[tst]]:
                f.write("./data/images/" + file + "\n")

        with open("src/data/filenames/images_val.txt", "w") as f:
            for file in np.asarray(self.images_files)[[val]]:
                f.write("./data/images/" + file + "\n")

        with open("src/data/filenames/images.txt", "w") as f:
            for file in np.asarray(self.images_files)[[idx]]:
                f.write("./data/images/" + file + "\n")

        # Landmark files
        with open("src/data/filenames/landmarks_test.txt", "w") as f:
            for file in np.asarray(self.landmark_files)[[tst]]:
                f.write("./data/landmarks/" + file + "\n")

        with open("src/data/filenames/landmarks_val.txt", "w") as f:
            for file in np.asarray(self.landmark_files)[[val]]:
                f.write("./data/landmarks/" + file + "\n")

        with open("src/data/filenames/landmarks.txt", "w") as f:
            for file in np.asarray(self.landmark_files)[[idx]]:
                f.write("./data/landmarks/" + file + "\n")

        return

    def SetIDs(self, labels):
        # ankle 0: Tibia --> 3: Tibia/Fibula
        # 1: Talus --> 4
        # 2: tendon --> 1

        for l in labels:
            label = open(dir+"labels/"+l, "r")
            landmark = open(dest+"landmarks/"+l, "w")
            for line in label:
                if line[0] == "0":
                    landmark.write("3" + line[1:])

                elif line[0] == "1":
                    landmark.write("4" + line[1:])

                elif line[0] == "2":
                    landmark.write("1" + line[1:])

            label.close()
            landmark.close()

        return


class ModelLog(object):
    def __init__(self):
        pass


def vizualize(path, target):
    """
        ._.
    """
    np_image = cv2.imread(path, cv2.IMREAD_COLOR)
    # (y: from up to down ,x)
    # transpose: (x: from right to left, y)
    pts = np.array([[0.5886666666666666, 0.3093869731800766],
                    [0.7573333333333332, 0.5114942528735632],
                    [0.2901135646687697, 0.7315857377433733]])
    width = np.array([0.2813333333333333, 0.16266666666666665, 0.3])
    height = np.array([0.1743295019157088, 0.17624521072796934, 0.1475095785440613])
    pts[:, 0] *= np_image.shape[1]
    pts[:, 1] *= np_image.shape[0]
    width *= np_image.shape[1]
    height *= np_image.shape[0]
    h, w = np_image.shape
    # for i in range(3):
    #     size = np.array([[(width[i])/2 + pts[i][0], (height[i])/2 + pts[i][1]]])
    #     pts = np.append(pts, size, axis=0)
    pts = np.floor(pts).astype(int)
    height = np.ceil(height).astype(int)
    width = np.ceil(width).astype(int)
    np_image = cv2.transpose(np_image)
    print(np_image.shape)
    for i in range(3):
        np_image = cv2.circle(np_image, (pts[i][1], pts[i][0]), radius=10, color=255)
    # np_image = cv2.circle(np_image, (pts[0][0] + width[0], pts[0][1] - height[0]), radius=6, color=(0, 0, 255))
    cv2.imshow("image", np_image.transpose(1, 0))
    cv2.waitKey(0)

    # landmarks = np.zeros((3, 2))
    # landmarks[:] = np.nan
    #
    # with open(target) as t:
    #     lines = [x.strip() for x in list(t) if x]
    #     for l in lines:
    #         info = l.split(" ")
    #         id = int(info[0])
    #         print(info[1:])
    #         landmarks[id, :] = info[1:3]
    #         print(landmarks)
    #
    # landmarks = np.asarray(landmarks)
    # landmarks = landmarks.reshape((-1, landmarks.shape[1]))

    return


def save_training(path, save_path):
    dist = {}
    score = {}
    epoch = {}
    validation = {}
    info_epoch = {}
    success_train = {}
    agents = {"0": 0, "1": 0, "2": 0}
    epoch_num = 1
    with open(path) as t:
        lines = [x.strip() for x in list(t) if x]
        for l in lines:
            info = l.split(" ")
            if info[0] == "train/dist" or info[0] == "train/score":
                ep = info[2].replace(":", "")
                dic = " ".join(info[3:]).replace("'", "\"")
                dic = json.loads(dic)
                dist[ep] = dic
                score[ep] = dic

            elif info[0] == "train":
                ep = info[2].replace(":", "")
                dic = " ".join(info[3:]).replace("'", "\"")
                dic = json.loads(dic)
                if not ep in info_epoch.keys():
                    info_epoch[ep] = {}

                for k in list(dic.keys()):
                    info_epoch[ep][k] = dic[k]
                epoch_num += 1

            elif info[0] == "train/mean_dist":
                ep = info[2].replace(":", "")
                dic = " ".join(info[3:]).replace("'", "\"")
                dic = json.loads(dic)
                agent = list(dic.keys())[0]
                if not ep in epoch.keys():
                    epoch[ep] = {}

                epoch[ep][agent] = dic[agent]

            elif info[0] == "eval/mean_dist":
                ep = info[2].replace(":", "")
                dic = " ".join(info[3:]).replace("'", "\"")
                dic = json.loads(dic)
                agent = list(dic.keys())[0]
                if not ep in validation.keys():
                    validation[ep] = {}

                validation[ep][agent] = dic[agent]

            elif info[0] == "distance":
                if not epoch_num in success_train.keys():
                    epoch_num -= 1
                    success_train[epoch_num] = copy.deepcopy(agents)
                success_train[epoch_num][info[3]] += 1

    with open(save_path + "train-dist.json", "w") as file:
        json.dump(dist, file)

    with open(save_path + "train-score.json", "w") as file:
        json.dump(score, file)

    with open(save_path + "train-epoch.json", "w") as file:
        json.dump(epoch, file)

    with open(save_path + "val-mean.json", "w") as file:
        json.dump(validation, file)

    with open(save_path + "info-epoch.json", "w") as file:
        json.dump(info_epoch, file)

    with open(save_path + "success-train.json", "w") as file:
        json.dump(success_train, file)

    return


def plot_successes(path):
    with open(path + "success-train.json", "r") as i:
        log = json.load(i)
    x = np.asarray(list(log.keys())).astype(int)
    success = np.asarray(list(log.values()))
    values0 = [d["0"] for d in success]
    values1 = [d["1"] for d in success]
    values2 = [d["2"] for d in success]

    plt.figure(1)
    plt.plot(x, values0, 'c-', label="Agent 0")
    plt.plot(x, values1, 'm-', label="Agent 1")
    plt.plot(x, values2, 'k-', label="Agent 2")
    plt.title("Number of Successes During Training")
    plt.xlabel("Epoch")
    plt.ylabel("# of Successes")
    plt.legend(loc='upper right')
    plt.text(0, 40, s="Agent 0 total successes:{}".format(sum(values0)))
    plt.text(0, 38, s="Agent 1 total successes:{}".format(sum(values1)))
    plt.text(0, 36, s="Agent 2 total successes:{}".format(sum(values2)))
    plt.show()

    return


def plot_log(train, val, name1, name2, save_path, agents=1):
    with open(train, "r") as t, open(val, "r") as v:
        dist = json.load(t)
        vdist = json.load(v)

    entries = dist.keys()
    agent = np.zeros((agents, len(entries)))
    vagent = np.zeros((agents, len(entries)))
    x = np.arange(len(entries))
    for i in range(agents):
        agent[i] = [d[str(i)] for d in list(dist.values())]
        vagent[i] = [d[str(i)] for d in list(vdist.values())]

    # plt.figure(1)
    plt.subplot(1, 2, 1)
    for i in range(agents):
        plt.plot(x, agent[i], 'c-', label="Agent {}".format(i))
    plt.title(name1)
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm)")
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    for i in range(agents):
        plt.plot(x, vagent[i], 'c-', label="Agent {}".format(i))
    plt.title(name2)
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm)")
    plt.legend(loc='upper right')

    plt.savefig(save_path + "results-dist.png")
    plt.show()

    return


def plot_loss(info):
    with open(info, "r") as i:
        log = json.load(i)

    x = np.arange(len(log.keys()))
    loss = [d["loss"] for d in list(log.values())]

    plt.figure(1)
    plt.plot(x, loss, 'k-', label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

    return


def read_output(file, agents=1):
    """
    ['number',
    'Filename 0', 'Agent 0 pos x', 'Agent 0 pos y', 'Landmark 0 pos x', 'Landmark 0 pos y', 'Distance 0',
    'Filename 1', 'Agent 1 pos x', 'Agent 1 pos y', 'Landmark 1 pos x', 'Landmark 1 pos y', 'Distance 1',
    'Filename 2', 'Agent 2 pos x', 'Agent 2 pos y', 'Landmark 2 pos x', 'Landmark 2 pos y', 'Distance 2']
    return:
    min distances = {file name: {agent #: [[agent x, agent y], [landmark x, landmark y]]}}
    max distances = {...}
    min_dist = [agent 0 min dist, agent 1 ..., ...]
    max_dist = [agent 0 max dist, agent 1 ..., ...]
    """
    with open(file) as f:
        lines = [x.split() for x in list(f) if x]

    labels = " ".join(lines[4]).replace("'", "\"")
    labels = json.loads(labels)

    dis_idx = []
    file_idx = []
    min_dist = []
    max_dist = []
    for i in range(agents):
        dis_idx.append(labels.index("Distance {}".format(i)))
        file_idx.append(labels.index("Filename {}".format(i)))
        min_dist.append(float('inf'))
        max_dist.append(0)

    file_min = {}
    file_max = {}
    for line in lines[5:-2]:
        l = " ".join(line).replace("'", "\"")
        l = json.loads(l)
        for i in range(agents):
            if l[dis_idx[i]] < min_dist[i]:
                min_dist[i] = l[dis_idx[i]]
                fidx = file_idx[i]
                coor = [l[fidx + 1], l[fidx + 2]]
                land = [l[fidx + 3], l[fidx + 4]]
                file_min["Agent {}".format(i)] = {l[fidx]: [coor, land]}

            if l[dis_idx[i]] > max_dist[i]:
                max_dist[i] = l[dis_idx[i]]
                fidx = file_idx[i]
                coor = [l[fidx + 1], l[fidx + 2]]
                land = [l[fidx + 3], l[fidx + 4]]
                file_max["Agent {}".format(i)] = {l[fidx]: [coor, land]}

    return file_min, file_max, min_dist, max_dist


def image_show(min_dic, max_dic, dmin, dmax):
    path = "src/data/images/"
    save = "src/tests/test-results/"
    size = (450, 450)

    for i in range(len(dmin)):
        min_name = list(min_dic["Agent {}".format(i)].keys())[0]
        max_name = list(max_dic["Agent {}".format(i)].keys())[0]

        im_min = cv2.imread(path + min_name + ".png")
        im_max = cv2.imread(path + max_name + ".png")

        coord = np.array(min_dic["Agent {}".format(i)][min_name])
        coordx = np.array(max_dic["Agent {}".format(i)][max_name])

        # Agent start = green, end = blue, Landmark = red
        im1 = cv2.circle(im_min, (round(0.5 * im_min.shape[1]), round(0.5 * im_min.shape[0])),
                         radius=3, color=(0, 255, 0), thickness=-1)
        im1 = cv2.circle(im_min, tuple(coord[0].astype(int)),
                         radius=4, color=(255, 0, 0), thickness=-1)
        im1 = cv2.circle(im_min, tuple(coord[1].astype(int)),
                         radius=4, color=(0, 0, 255), thickness=-1)
        ######
        im2 = cv2.circle(im_max, (round(0.5 * im_max.shape[1]), round(0.5 * im_max.shape[0])),
                         radius=4, color=(0, 255, 0), thickness=-1)
        im2 = cv2.circle(im_max, tuple(coordx[0].astype(int)),
                         radius=4, color=(255, 0, 0), thickness=-1)
        im2 = cv2.circle(im_max, tuple(coordx[1].astype(int)),
                         radius=4, color=(0, 0, 255), thickness=-1)

        im1 = cv2.resize(im1, size)
        im2 = cv2.resize(im2, size)
        cv2.putText(im1, "Agent {}: Distance From Landmark {}mm".format(i, round(dmin[i])), (20, 420),
                    thickness=1, fontScale=0.4, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(im2, "Agent {}: Distance From Landmark {}mm".format(i, round(dmax[i])), (20, 420),
                    thickness=1, fontScale=0.4, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        himage = np.hstack((im1, im2))

        cv2.imwrite(save + "Agent{}-Tendon".format(i) + ".png", himage)
    # cv2.imshow("Test: Min/Max Distance Image", himage)
    # cv2.waitKey(0)

    return


if __name__ == '__main__':

    dir = "/Users/dianaescoboza/Documents/SUMMER22/Datasets/KneeDS/images/"
    dest = "/Users/dianaescoboza/Documents/SUMMER22/Datasets/Datasets/images/"

    # data = CleanData(dir, "src/data/landmarks/")
    # data.ImageNorm(dir, dest)
    # data.ModelFilenames()
    # data.RescaleImages()

    # vizualize(im, lan)
    # logs = "src/test-sync/myserver/logs.txt"
    # train_dist = "src/test-sync/myserver/train-epoch.json"
    # val_dist = "src/test-sync/myserver/val-mean.json"
    # save_dir = "src/test-sync/myserver/"
    # save_training(logs, save_dir)
    # plot_log(train_dist, val_dist,
    #         "Train Mean Distance (new loss)", "Validation Mean Distance (new loss)", save_dir, agents=1)
    # plot_loss("src/tests/test-results/info-epoch.json")

    # dic_min, dic_max, min_dist, max_dist = read_output("src/runs/Jul20_13-17-34_MacBook-Pro.local/logs.txt")
    # image_show(dic_min, dic_max, min_dist, max_dist)
