import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd


def imageNorm(path, destination):

    image_files = [f for f in os.listdir(path) if not f.startswith('.')]
    idx = 0
    for im in image_files:
        image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Image is empty: {}, {}".format(idx, im))
        im_norm = np.zeros(image.shape)
        im_norm = cv2.normalize(image, im_norm, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(destination + im, im_norm)
        idx += 1

    return

def vizualize(path, target):
    """
        ._.
    """
    np_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

def save_training(path):

    dist = {}
    score = {}
    epoch = {}
    validation = {}
    info_epoch = {}
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

    with open("src/results-2a/train-dist.json", "w") as file:
        json.dump(dist, file)

    with open("src/results-2a/train-score.json", "w") as file:
        json.dump(score, file)

    with open("src/results-2a/train-epoch.json", "w") as file:
        json.dump(epoch, file)

    with open("src/results-2a/val-mean.json", "w") as file:
        json.dump(validation, file)

    with open("src/results-2a/info-epoch.json", "w") as file:
        json.dump(info_epoch, file)

    return

def plot_log(train, val,name1,name2):

    with open(train, "r") as t, open(val, "r") as v:
        dist = json.load(t)
        vdist = json.load(v)

    x = list(dist.keys())
    agent0 = [d["0"] for d in list(dist.values())]
    agent1 = [d["1"] for d in list(dist.values())]
    #agent2 = [d["2"] for d in list(dist.values())]

    vagent0 = [d["0"] for d in list(vdist.values())]
    vagent1 = [d["1"] for d in list(vdist.values())]
    #vagent2 = [d["2"] for d in list(vdist.values())]

    #plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(x, agent0, label="Agent 0")
    plt.plot(x, agent1, label="Agent 1")
    # plt.plot(x, agent2, label="Agent 2")
    plt.title(name1)
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm)")

    plt.subplot(1, 2, 2)
    plt.plot(x, vagent0, label="Agent 0")
    plt.plot(x, vagent1, label="Agent 1")
    # plt.plot(x, vagent2, label="Agent 2")
    plt.title(name2)
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm)")

    plt.show()

    return

def plot_loss(info):

    with open(info, "r") as i:
        log = json.load(i)

    episode = list(log.keys())
    loss = [d["loss"] for d in list(log.values())]

    plt.figure(1)
    plt.plot(episode, loss, label="Loss")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

    return

if __name__ == '__main__':
    # dir = "src/data/images/"
    # dest = "src/data/norm_images/"
    # im = "src/data/images/0a63a82d-4833___m4732_a4833_s4873_1_8_US_.png"
    # lan = "src/data/landmarks/0a63a82d-4833___m4732_a4833_s4873_1_8_US_.txt"
    # imageNorm(path=dir, destination=dest)
    #vizualize(im, lan)

    logs = "src/test-sync/myserver/logs.txt"
    train_dist = "src/results-2a/train-epoch.json"
    val_dist = "src/results-2a/val-mean.json"
    #save_training(logs)
    # plot_log(train_dist, val_dist,
    #          "Train Mean Distance", "Validation Mean Distance")

    plot_loss("src/results-2a/info-epoch.json")




