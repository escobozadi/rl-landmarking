import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import copy


class CleanData(object):
    def __init__(self, dir=None, landmarks_dir=None, land_dest=None, num_ds=3):
        self.images_dir = dir
        self.ds = num_ds
        self.labels = landmarks_dir
        self.landmarks_dest = land_dest
        # self.images_files = [f for f in os.listdir(dir) if not f.startswith('.')]
        # self.landmark_files = [f for f in os.listdir(landmarks_dir) if not f.startswith('.')]

    def ImageNorm(self, path, destination, files):

        idx = 0
        for im in files:
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

    def ModelFilenames(self, knee, elbow, ankle):
        '''
        Separate in equal parts the datasets into train/val/test
        '''
        knee_files = np.array([f[:-4] for f in os.listdir(knee) if not f.startswith('.')])
        elbow_files = np.array([f[:-4] for f in os.listdir(elbow) if not f.startswith('.')])
        ankle_files = np.array([f[:-4] for f in os.listdir(ankle) if not f.startswith('.')])

        train_files = np.array([])
        val_files = np.array([])
        test_files = np.array([])

        size = [len(knee_files), len(elbow_files), len(ankle_files)]
        # 80/10/10 -> train/val/test
        portion = 0.1
        for i in range(self.ds):
            print(i)
            idx = np.arange(size[i])
            val = np.random.choice(idx, size=int(len(idx) * portion)).astype(int)
            tst = np.random.choice(np.delete(idx, val), size=int(len(idx) * portion)).astype(int)
            idx = np.delete(idx, np.append(tst, val)).astype(int)
            if i == 0:
                train_files = np.append(train_files, knee_files[idx])
                val_files = np.append(val_files, knee_files[val])
                test_files = np.append(test_files, knee_files[tst])
            elif i == 1:
                train_files = np.append(train_files, elbow_files[idx])
                val_files = np.append(val_files, elbow_files[val])
                test_files = np.append(test_files, elbow_files[tst])

            elif i == 2:
                train_files = np.append(train_files, ankle_files[idx])
                val_files = np.append(val_files, ankle_files[val])
                test_files = np.append(test_files, ankle_files[tst])
        # shuffle
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)
        np.random.shuffle(test_files)
        print(test_files)

        # TEST FILENAMES
        with open("src/data/filenames/images_test.txt", "w") as f:
            for file in test_files:
                f.write("./data/images/" + file + ".png" + "\n")

        with open("src/data/filenames/landmarks_test.txt", "w") as f:
            for file in test_files:
                f.write("./data/landmarks/" + file + ".txt" + "\n")


        # VAL FILENAMES
        with open("src/data/filenames/images_val.txt", "w") as f:
            for file in val_files:
                f.write("./data/images/" + file + ".png" + "\n")

        with open("src/data/filenames/landmarks_val.txt", "w") as f:
            for file in val_files:
                f.write("./data/landmarks/" + file + ".txt" + "\n")


        # TRAIN FILENAMES
        with open("src/data/filenames/images.txt", "w") as f:
            for file in train_files:
                f.write("./data/images/" + file + ".png" + "\n")

        with open("src/data/filenames/landmarks.txt", "w") as f:
            for file in train_files:
                f.write("./data/landmarks/" + file + ".txt" + "\n")

        return

    def SetIDs(self, dataset=None):
        # ankle
        # 0: Tibia --> 3: Tibia/Fibula
        # 1: Talus --> 4
        # 2: tendon --> 1
        # elbow,
        # 0: Ulna --> 5
        # 1: triceps tendon insertion --> 6
        # 2: humerus --> 7
        # 3: quadriceps tendon --> 1
        if dataset == "ankle":
            new_ids = ["3", "4", "1"]
        else:  # dataset == "elbow":
            new_ids = ["5", "6", "7", "1"]

        for l in self.landmark_files:
            label = open(self.labels + l, "r")
            landmark = open(self.landmarks_dest + l, "w")
            for line in label:
                if line[0] == "0":
                    landmark.write(new_ids[0] + line[1:])

                elif line[0] == "1":
                    landmark.write(new_ids[1] + line[1:])

                elif line[0] == "2":
                    landmark.write(new_ids[2] + line[1:])

                elif line[0] == "3":
                    landmark.write(new_ids[3] + line[1:])

            label.close()
            landmark.close()

        return

    def AvgBoxes(self):

        for file in self.landmark_files:
            with open(self.labels + file, "r") as target:
                landmark = [line.strip() for line in list(target)]
            target.close()
            dic = {}
            for l in landmark:
                info = l.split(" ")
                id = int(info[0])
                if id not in list(dic.keys()):
                    dic[id] = []
                    dic[id].append(info[1:])
                else:
                    dic[id].append(info[1:])

            for entry in list(dic.keys()):
                dic[entry] = np.asarray(dic[entry])
                if len(dic[entry]) > 1:
                    mean = np.asarray([])
                    for i in range(len(dic[entry][0])):
                        mean = np.append(mean, np.mean(dic[entry][:, i].astype(float)))
                    dic[entry] = np.array([mean]).astype(str)

            with open(self.landmarks_dest + file, "w") as new:
                for k in list(dic.keys()):
                    new.write(str(k) + " ")
                    new.write(' '.join(dic[k][0]))
                    new.write("\n")
            new.close()

        return


class ModelLog(object):
    def __init__(self):
        pass

    def save_training(self, path, save_path):

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
                    dic = " ".join(info[3:]).replace("'", "\"").replace("nan", "NaN")
                    dic = json.loads(dic)
                    dist[ep] = dic
                    score[ep] = dic

                elif info[0] == "train":
                    ep = info[2].replace(":", "")
                    dic = " ".join(info[3:]).replace("'", "\"").replace("nan", "NaN")
                    dic = json.loads(dic)
                    if not ep in info_epoch.keys():
                        info_epoch[ep] = {}

                    for k in list(dic.keys()):
                        info_epoch[ep][k] = dic[k]
                    epoch_num += 1

                elif info[0] == "train/mean_dist":
                    ep = info[2].replace(":", "")
                    dic = " ".join(info[3:]).replace("'", "\"").replace("nan", "NaN")
                    dic = json.loads(dic)
                    agent = list(dic.keys())[0]
                    if not ep in epoch.keys():
                        epoch[ep] = {}

                    epoch[ep][agent] = dic[agent]

                elif info[0] == "eval/mean_dist":
                    ep = info[2].replace(":", "")
                    dic = " ".join(info[3:]).replace("'", "\"").replace("nan", "NaN")
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

    def saveBaseline(self, path):
        log = path + "/logs.txt"
        train_loss = {}
        train_dists = {}
        val_loss = {}
        val_dists = {}

        type = "train"
        epoch = 0
        with open(log) as t:
            lines = [x.strip().split(" ") for x in list(t) if x]
        for line in lines:
            if line[0] == "EPOCH":
                epoch = int(line[1])
                if epoch not in train_loss:
                    train_loss[epoch] = []
                    train_dists[epoch] = []
                else:
                    val_loss[epoch] = []
                    val_dists[epoch] = []
                continue
            elif line[1] == "Total":
                if line[0] == "Train":
                    train_loss[epoch].append(line[3])
                    type = "train"
                elif line[0] == "Validation":
                    val_loss[epoch].append(line[3])
                    type = "val"
                continue
            elif line[0] == "Target":
                if type == "train":
                    train_loss[epoch].append(line[3][:-1])
                    train_loss[epoch].append(line[6])
                else:
                    val_loss[epoch].append(line[3][:-1])
                    val_loss[epoch].append(line[6])
                continue
            elif line[0] == "Agent":
                if type == "train":
                    train_dists[epoch].append(line[2])
                    train_dists[epoch].append(line[6])
                    train_dists[epoch].append(line[10])
                    train_dists[epoch].append(line[14])
                    train_dists[epoch].append(line[18])
                    train_dists[epoch].append(line[22])
                    train_dists[epoch].append(line[26])
                    train_dists[epoch].append(line[30])
                else:
                    val_dists[epoch].append(line[2])
                    val_dists[epoch].append(line[6])
                    val_dists[epoch].append(line[10])
                    val_dists[epoch].append(line[14])
                    val_dists[epoch].append(line[18])
                    val_dists[epoch].append(line[22])
                    val_dists[epoch].append(line[26])
                    val_dists[epoch].append(line[30])
                continue

        with open(path + "/train-dists.json", "w") as file:
            json.dump(train_dists, file)

        with open(path + "/val-dists.json", "w") as file:
            json.dump(val_dists, file)

        with open(path + "/train-loss.json", "w") as file:
            json.dump(train_loss, file)

        with open(path + "/val-loss.json", "w") as file:
            json.dump(val_loss, file)

        return

    def read_output(self, file, agents=3):
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
        print(labels)
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
            l = " ".join(line).replace("'", "\"").replace("nan", "\"nan\"")
            l = json.loads(l)
            for i in range(agents):
                if l[dis_idx[i]] == "nan":
                    continue

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

    def image_show(self, min_dic, max_dic, dmin, dmax, save="src/runs/ankle-test-preliminary/", joint="ankle"):
        path = "src/data/images/"
        if joint == "ankle":
            landmarks = ["Tibia/Fibula", "Talus", "Quadriceps Tendon"]
        elif joint == "elbow":
            landmarks = ["Quadriceps Tendon", "Humerus", "Ulna"]
        else:   # knee
            landmarks = ["Femur", "Quadriceps Tendon", "Patella"]
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

            cv2.imwrite(save + "Agent{}-{}".format(i, landmarks[i]) + ".png", himage)
        cv2.imshow("Test: Min/Max Distance Image", himage)
        cv2.waitKey(0)

        return


class plotTraining(object):
    pass

    def vizualize(self, path, target):
        """
            ._.
        """
        label = "a60fc140-1498___m1488_a1498_s1511_0_186050_US_.txt"
        image = cv2.imread(dir + label[:-4] + ".png")
        with open(elbow_landmarks + label, "r") as target:
            landmark = [line.strip() for line in list(target)]
        target.close()
        dic = {}
        for l in landmark:
            info = l.split(" ")
            id = int(info[0])
            if id not in list(dic.keys()):
                dic[id] = []
                dic[id].append(info[1:])
            else:
                dic[id].append(info[1:])

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

    def plot_successes(self, path):
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

    def plot_log(self, train, val, name1, name2, save_path, agents=1):
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

        colors = ["c-", "b-", "g-", "r-", "m-", "y-", "k-", "k-", "b-", "g-"]
        # plt.figure(1)
        plt.subplot(1, 2, 1)
        for i in range(agents):
            plt.plot(x, agent[i], colors[i], label="Agent {}".format(i))
        plt.title(name1)
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        for i in range(agents):
            plt.plot(x, vagent[i], colors[i], label="Agent {}".format(i))
        plt.title(name2)
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.savefig(save_path + "results-dist.png")
        plt.show()

        return

    def plot_loss(self, info):
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

    def plot_baseline(self, path):
        train_dists = json.load(open(path + "/train-dists.json", "r"))
        val_dists = json.load(open(path + "/val-dists.json", "r"))
        train_loss = json.load(open(path + "/train-loss.json", "r"))
        val_loss = json.load(open(path + "/val-loss.json", "r"))

        epochs = np.asarray(list(train_loss.keys())).astype(int)
        agent = np.zeros((8, len(epochs)))
        vagent = np.zeros((8, len(epochs)))
        tdists = np.asarray(list(train_dists.values())).astype(float)
        vdists = np.asarray(list(val_dists.values())).astype(float)
        for i in range(8):
            agent[i] = tdists[:, i]
            vagent[i] = vdists[:, i]
        tloss = np.asarray(list(train_loss.values())).astype(float)
        vloss = np.asarray(list(val_loss.values())).astype(float)

        colors = ["k-", "c-", "b-", "g-", "r-", "m-", "y-", "k-"]
        plt.subplot(2, 3, 1)
        plt.plot(epochs, tloss[:, 0], "g-", label="Mean Total Loss")
        # plt.plot(epochs, vloss[:, 0], "r-", label="Val Total Loss")
        plt.title("Total Mean Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 2)
        plt.plot(epochs, tloss[:, 1], "g-", label="Train Mean Distance Loss")
        # plt.plot(epochs, vloss[:, 1], "r-", label="Val Mean Distance Loss")
        plt.title("Distance to Target Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 4)
        plt.plot(epochs, tloss[:, 2], "g-", label="Mean Class Loss")
        # plt.plot(epochs, vloss[:, 2], "r-", label="Val Class Loss")
        plt.title("Landmark Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 5)
        plt.plot(epochs, agent[0], "g-", label="Agent 0 Distance to Target")
        plt.title("Agent 0")
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 3)
        for i in range(8):
            plt.plot(epochs, agent[i], colors[i], label="Target {}".format(i))
        plt.title("Distance Target Training")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.subplot(2, 3, 6)
        for i in range(8):
            plt.plot(epochs, vagent[i], colors[i], label="Target {}".format(i))
        plt.title("Distance Target Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend(loc='upper right')

        plt.savefig(path + "/results.png")
        plt.show()
        return


def images_names(directory, destination, im_files):

    for file in im_files:
        image = cv2.imread(directory + file)
        if file[-12:] == "_cleanup.png":
            name = file[:-12] + ".png"
        else:
            name = file
        cv2.imwrite(destination + name, image)

    return


class FilesOrdering(object):
    def __init__(self, files):
        self.train_files = [f[1:-1] for f in open(files + "images.txt", "r")]
        self.val_files = [f[1:-1] for f in open(files + "images_val.txt", "r")]
        self.test_files = [f[1:-1] for f in open(files + "images_test.txt", "r")]
        self.labels_train = "landmarks.txt"
        self.labels_val = "landmarks_val.txt"
        self.labels_est = "landmarks_test.txt"
        self.train_dir = "./src/data/train/"
        self.val_dir = "./src/data/val/"
        self.test_dir = "./src/data/test/"
        self.files = files
        return

    def orderImages(self):

        for file in self.train_files:
            image = cv2.imread("./src" + file)
            name = file[13:]
            cv2.imwrite(self.train_dir + name, image)

        for file in self.val_files:
            image = cv2.imread("./src" + file)
            name = file[13:]
            cv2.imwrite(self.val_dir + name, image)

        for file in self.test_files:
            image = cv2.imread("./src" + file)
            name = file[13:]
            cv2.imwrite(self.test_dir + name, image)

        return

    def updateDir(self):
        train = "./data/train/"
        val = "./data/val/"
        test = "./data/test/"

        # TEST
        with open(self.files + "images_test2.txt", "w") as image, open(self.files + "landmarks_test2.txt", "w") as label:
            for file in self.test_files:
                image.write(test + file[13:])
                image.write("\n")
                label.write("./data/landmarks/" + file[13:-4] + ".txt")
                label.write("\n")

        # VAL
        with open(self.files + "images_val2.txt", "w") as image, open(self.files + "landmarks_val2.txt", "w") as label:
            for file in self.val_files:
                image.write(val + file[13:])
                image.write("\n")
                label.write("./data/landmarks/" + file[13:-4] + ".txt")
                label.write("\n")

        # TRAIN
        with open(self.files + "images2.txt", "w") as image, open(self.files + "landmarks2.txt", "w") as label:
            for file in self.train_files:
                image.write(train + file[13:])
                image.write("\n")
                label.write("./data/landmarks/" + file[13:-4] + ".txt")
                label.write("\n")

        image.close()
        label.close()

        return


def datasetOrder():
    dir = "/Users/dianaescoboza/Documents/SUMMER22/Datasets/KneeDS/knee-images"
    dest = "/Users/dianaescoboza/Documents/PycharmProjects/rl-landmark/rl-medical/src/data"
    files = np.array([f for f in os.listdir(dir) if f[0] != "."])
    n = files.shape[0]
    v = int(n * 0.2)
    t = int(n * 0.1)
    idxes = np.arange(n)
    validx = np.random.choice(idxes, v)
    tstidx = np.random.choice(np.delete(idxes, validx), t)
    idxes = np.delete(idxes, np.append(validx, tstidx))
    for im in files[tstidx]:
        image = cv2.imread(dir + "/" + im)
        image = cv2.resize(image, (256, 256))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dest + "/knee-test/" + im, image)

    for im in files[validx]:
        image = cv2.resize(cv2.imread(dir + "/" + im), (256, 256))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dest + "/knee-val/" + im, image)

    for im in files[idxes]:
        image = cv2.resize(cv2.imread(dir + "/" + im), (256, 256))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dest + "/knee-train/" + im, image)
    return


def datasetfilenames():
    train = "./src/data/knee-train"
    val = "./src/data/knee-val"
    tst = "./src/data/knee-test"
    train_label = [f for f in os.listdir(train) if f[0] != "."]
    val_label = [f for f in os.listdir(val) if f[0] != "."]
    tst_label = [f for f in os.listdir(tst) if f[0] != "."]

    with open("./src/data/knee-filenames/images_train.txt", "w") as f:
        for im in train_label:
            f.write("./src/data/knee-train/" + im + '\n')

    with open("./src/data/knee-filenames/images_val.txt", "w") as f:
        for im in val_label:
            f.write("./src/data/knee-val/" + im + '\n')

    with open("./src/data/knee-filenames/images_test.txt", "w") as f:
        for im in tst_label:
            f.write("./src/data/knee-test/" + im + '\n')
    return


def Annotations():
    knee_files = np.array([f for f in os.listdir(knee_dir) if f[0] != "."])
    knee_annotations = {}
    knee_annotations["categories"] = [{"id": 0, "name": "Femur"},
                                      {"id": 1, "name": "Quadriceps Tendon"},
                                      {"id": 2, "name": "Patella"}]
    n = 0
    knee_annotations["images"] = []
    knee_annotations["annotations"] = []
    for i in range(len(knee_files)):
        im = {}
        ann = {}
        im["id"] = i
        im["license"] = 1
        im["file_name"] = knee_files[i]
        image = cv2.imread(knee_dir + "/" + knee_files[i])
        height = image.shape[0]
        width = image.shape[1]
        im["height"] = height
        im["width"] = width
        knee_annotations["images"].append(im)
        ann["image_id"] = i
        ann["segmentation"] = []
        ann["iscrowd"] = 0
        with open(knee_label + "/" + knee_files[i][:-3] + "txt", "r") as t:
            lines = [x.strip() for x in list(t) if x]
            for line in lines:
                l = np.asarray(line.split(" ")).astype(float)
                ann["category_id"] = int(l[0])
                ann["bbox"] = [round(l[1] * width), round(l[2] * height),
                               round(l[3] * height), round(l[4] * width)]
                ann["area"] = round(l[3] * height) * round(l[4] * width)
                ann["id"] = n
                knee_annotations["annotations"].append(copy.deepcopy(ann))
                n += 1

    with open("/Users/dianaescoboza/Documents/PycharmProjects/rl-landmark/rl-medical/src/data/knee_test.json", "w") as j:
        json.dump(knee_annotations, j)
    return


if __name__ == '__main__':

    rl_ankle_dir = "./src/runs/ankle-test-preliminary/logs.txt"
    data = ModelLog()
    file_min, file_max, min_dist, max_dist = data.read_output(rl_ankle_dir)
    print(file_min)
    print(file_max)
    print(min_dist)
    print(max_dist)
    # data.image_show(file_min, file_max, min_dist, max_dist)








