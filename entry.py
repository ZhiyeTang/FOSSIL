import importlib
import json
import os
import numpy as np
from tqdm import tqdm

config = json.load(open("configs/Fossil.json"))


class Runner:
    def __init__(self, config):
        self.config = config
        self.epochs = config["epochs"]
        Algorithm_Class = importlib.import_module(
            "algorithm."+config["algorithm"]["name"])
        self.algorithm = Algorithm_Class.Algorithm(config["algorithm"])
        self._data_preprocess(
            config["user_num"], config["item_num"], config["T"])

        self.train_trace = []

    def _data_preprocess(self, user_num, item_num, T):
        # if data haven't been pre-processed (convert into matrices)
        if not os.path.exists("np_dataset"):
            os.makedirs("np_dataset")
        if not os.path.exists("np_dataset/train_data.dat"):
            train_data_raw = np.loadtxt(
                "dataset/Foursquare/Foursquare_train.csv", delimiter=",")
            valid_data_raw = np.loadtxt(
                "dataset/Foursquare/Foursquare_valid.csv", delimiter=",")
            test_data_raw = np.loadtxt(
                "dataset/Foursquare/Foursquare_test.csv", delimiter=",")
            data_raw = np.concatenate(
                [train_data_raw, valid_data_raw, test_data_raw], axis=0)
            nega_data_raw = np.loadtxt(
                "dataset/Foursquare/Foursquare_negative.csv", delimiter=",")
            nega_data_raw = nega_data_raw[nega_data_raw[:, 0].argsort()]
            nega_data_raw = nega_data_raw[:, 1:] - 1

            data = np.ones([user_num, item_num]) * np.inf
            for idx in range(data_raw.shape[0]):
                u = int(data_raw[idx, 0] - 1)
                i = int(data_raw[idx, 1] - 1)
                data[u, i] = data_raw[idx, 2]

            bak = data.copy()
            data = data.argsort(axis=1).argsort(axis=1)
            data[bak == np.inf] = -1
            data = data.astype(np.int32)
            data_list = []
            nega_data_list = []
            for u in range(user_num):
                if ((data[u, :] != -1).sum()) > 4:
                    data_list.append(data[u, :])
                    nega_data_list.append(nega_data_raw[u, :])
            data = np.concatenate(data_list).reshape([-1, item_num])
            nega_data = np.concatenate(nega_data_list).reshape([-1, 100])
            user_num = data.shape[0]

            train_data = -np.ones([user_num, T]).astype(np.int32)
            for u in tqdm(range(user_num), desc="Pre-Processing Data"):
                for i in range(item_num):
                    if not data[u, i] == -1:
                        train_data[u, data[u, i]] = i

            valid_data = data.argmax(axis=1)
            train_data[np.arange(user_num),
                       data[np.arange(user_num), valid_data]] = -1

            np.savetxt("np_dataset/train_data.dat", train_data)
            np.savetxt("np_dataset/valid_data.dat", valid_data)
            np.savetxt("np_dataset/nega_data.dat", nega_data)

        # load the pre-processed data (as matrices)
        self.train_data = np.loadtxt(
            "np_dataset/train_data.dat").astype(np.int32)
        self.valid_data = np.loadtxt(
            "np_dataset/valid_data.dat").astype(np.int32)
        self.nega_data = np.loadtxt(
            "np_dataset/nega_data.dat").astype(np.int32)

    def train(self):
        for step in range(self.epochs):
            # train
            for user_id in range(self.train_data.shape[0]):
                neg_idx = np.random.randint(100)
                pos_idx = np.random.randint(
                    config["algorithm"]["L"],
                    (self.train_data[user_id, :] != -1).sum()
                )
                self.algorithm.train(
                    user_id,
                    self.train_data[user_id, :],
                    self.nega_data[user_id, neg_idx],
                    pos_idx,
                )
            if step % 10 == 9:
                RecK = self.algorithm.eval(
                    self.train_data,
                    self.valid_data,
                    # user_id,
                    # self.train_data[user_id, :],
                    # self.nega_data[user_id, neg_idx],
                    # pos_idx,
                )
                self.train_trace.append(RecK)
                print(
                    "Recall@K[epoch{}]: {:.8}%".format(str(step+1).zfill(3), RecK))
        self.save()

    def save(self):
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        if not os.path.exists("checkpoint/"+self.config["algorithm"]["name"]):
            os.makedirs("checkpoint/"+self.config["algorithm"]["name"])
        self.algorithm.save("checkpoint/"+self.config["algorithm"]["name"])
        np.savetxt(
            os.path.join(
                "checkpoint/"+self.config["algorithm"]["name"], "trace.txt"),
            np.array(self.train_trace, dtype=np.float32)
        )


if __name__ == "__main__":
    runner = Runner(config)
    runner.train()
