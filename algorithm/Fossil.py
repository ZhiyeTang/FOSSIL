import os
from tqdm import tqdm
import numpy as np


class Algorithm:
    def __init__(self, config):
        self.d = config["d"]
        self.UserNum = config["UserNum"]
        self.ItemNum = config["ItemNum"]
        self.L = config["L"]
        self.K = config["K"]

        self.V = np.random.normal(0, .001, size=[self.ItemNum, self.d])
        self.W = np.random.normal(0, .001, size=[self.ItemNum, self.d])
        self.b = np.random.normal(0, .001, size=[self.ItemNum])
        self.etaU = np.random.normal(0, .001, size=[self.UserNum, self.L])
        self.eta = np.random.normal(0, .001, size=[self.L])

        self.alpha = config["alpha"]
        self.gamma = config["gamma"]

    def train(self, user_id: int, user_items: np.ndarray, neg_item: int, t: int) -> None:

        # Using long term knowledge and short term knowledge to estimate scores of positive item
        # and negative item.
        pos_item = user_items[t]
        user_items = user_items[:t]
        longterm = np.zeros(self.d)
        for i in range(t):
            longterm += self.W[user_items[i], :]
        longterm /= np.sqrt(t)

        shortterm = np.zeros(self.d)
        for l in range(self.L):
            shortterm += (self.eta[l] + self.etaU[user_id, l]
                          ) * self.W[user_items[-l - 1], :]

        U = longterm + shortterm
        R_pos = self.b[pos_item] + np.dot(U, self.V[pos_item, :])
        R_neg = self.b[neg_item] + np.dot(U, self.V[neg_item, :])

        # Compute gradients of each parameters that involved.
        coef = 1 / (1 + np.exp(-np.clip((R_neg - R_pos), -10, 10)))
        gradb_pos = self.alpha * self.b[pos_item] - coef
        gradb_neg = self.alpha * self.b[neg_item] + coef
        gradV_pos = self.alpha * self.V[pos_item, :] - coef * U
        gradV_neg = self.alpha * self.V[neg_item, :] + coef * U
        gradeta = self.alpha * self.eta - \
            coef * np.dot(
                self.W[user_items[-1: -self.L - 1: -1], :],
                self.V[pos_item, :] - self.V[neg_item, :],
            )
        gradetaU = self.alpha * self.etaU[user_id, :] - \
            coef * np.dot(
                self.W[user_items[-1: -self.L - 1: -1], :],
                self.V[pos_item, :] - self.V[neg_item, :],
        )
        gradW_pos = self.alpha * self.W[pos_item, :] - \
            coef * (self.V[pos_item] / np.sqrt(t - 1) -
                    self.V[neg_item] / np.sqrt(t))
        gradW_neg = self.alpha * self.W[neg_item, :] - \
            coef * (-self.V[neg_item] / np.sqrt(t))
        # gradW_shortterm = self.alpha * self.W[user_items[-1: -self.L - 1: -1], :] - \
        #     coef * np.outer(
        #         self.eta + self.etaU[user_id, :],
        #         self.V[pos_item, :] - self.V[neg_item, :],
        #     ) - \
        #     coef * (self.V[pos_item] / np.sqrt(t - 1) -
        #             self.V[neg_item] / np.sqrt(t))
        gradW_shortterm = -coef * np.outer(
                self.eta + self.etaU[user_id, :],
                self.V[pos_item, :] - self.V[neg_item, :],
            )

        # Update parameters.
        self.b[pos_item] -= self.gamma * gradb_pos
        self.b[neg_item] -= self.gamma * gradb_neg
        self.V[pos_item, :] -= self.gamma * gradV_pos
        self.V[neg_item, :] -= self.gamma * gradV_neg
        self.eta -= self.gamma * gradeta
        self.etaU -= self.gamma * gradetaU
        self.W[pos_item, :] -= self.gamma * gradW_pos
        self.W[neg_item, :] -= self.gamma * gradW_neg
        self.W[user_items[-1: -self.L - 1: -1],
               :] -= self.gamma * gradW_shortterm

    def eval(self, train_data: np.ndarray, valid_data: np.ndarray) -> float:

        T = (train_data != -1).sum(axis=1)
        Recall_at_K = 0
        for user_id in range(self.UserNum):
            t = T[user_id]
            user_items = train_data[user_id, :t]
            R = np.zeros([self.ItemNum])

            longterm = np.zeros(self.d)
            for i in range(t):
                longterm += self.W[user_items[i], :]
                R[user_items[i]] = -np.inf
            longterm /= np.sqrt(t)

            shortterm = np.zeros(self.d)
            for l in range(self.L):
                shortterm += (self.eta[l] + self.etaU[user_id, l]
                              ) * self.W[user_items[-l - 1], :]

            U = longterm + shortterm

            for item_id in range(self.ItemNum):
                if R[item_id] == 0:
                    R[item_id] = self.b[item_id] + \
                        np.dot(U, self.V[item_id, :])

            Recall = (-R).argsort()[:self.K]
            if valid_data[user_id] in Recall:
                Recall_at_K += 1

        return Recall_at_K / self.UserNum

    # def eval(self, user_id: int, user_items: np.ndarray, neg_item: int, t: int) -> float:
    #     pos_item = user_items[t]
    #     user_items = user_items[:t]
    #     longterm = np.zeros(self.d)
    #     for i in range(t):
    #         longterm += self.W[user_items[i], :]
    #     longterm /= np.sqrt(t)

    #     shortterm = np.zeros(self.d)
    #     for l in range(self.L):
    #         shortterm += (self.eta[l] + self.etaU[user_id, l]
    #                       ) * self.W[user_items[- l - 1], :]

    #     U = longterm + shortterm
    #     R_pos = self.b[pos_item] + np.dot(U, self.V[pos_item, :])
    #     R_neg = self.b[neg_item] + np.dot(U, self.V[neg_item, :])

    #     reg = np.square(self.V[pos_item]).sum() + \
    #         np.square(self.V[neg_item]).sum() + \
    #         np.square(longterm * np.sqrt(t)).sum() + \
    #         np.square(self.b[pos_item]) + \
    #         np.square(self.b[neg_item]) + \
    #         np.square(self.eta).sum() + \
    #         np.square(self.etaU[user_id]).sum()

    #     return -np.log(1 / (1 + np.exp(-np.clip((R_pos - R_neg), -10, 10)))).sum() + \
    #         0.5 * self.alpha * reg

    def save(self, path):
        np.savetxt(os.path.join(path, "V.txt"), self.V)
        np.savetxt(os.path.join(path, "W.txt"), self.W)
        np.savetxt(os.path.join(path, "b.txt"), self.b)
        np.savetxt(os.path.join(path, "etaU.txt"), self.etaU)
        np.savetxt(os.path.join(path, "eta.txt"), self.eta)

    def load(self, path):
        self.V = np.loadtxt(os.path.join(path, "V.txt"))
        self.W = np.loadtxt(os.path.join(path, "W.txt"))
        self.b = np.loadtxt(os.path.join(path, "b.txt"))
        self.etaU = np.loadtxt(os.path.join(path, "etaU.txt"))
        self.eta = np.loadtxt(os.path.join(path, "eta.txt"))
