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

        self.loss = 0.

    def train(self, user_id: int, user_items: np.ndarray, pos_items: np.ndarray, neg_items: np.ndarray) -> None:

        # Using long term knowledge and short term knowledge to estimate scores of positive item
        # and negative item.
        for idx in range(len(pos_items)):
            t = self.L + idx
            user_items_t = user_items[:t]
            longterm = self.W[user_items_t, :].sum(axis=0) / np.sqrt(t)
            shortterm = np.matmul(
                self.eta + self.etaU[user_id, :], self.W[user_items[:-self.L-1:-1], :])
            U = longterm + shortterm

            R_pos = self.b[pos_items[idx]] + np.dot(U, self.V[pos_items[idx], :])
            R_neg = self.b[neg_items[idx]] + np.dot(U, self.V[neg_items[idx], :])

            Reg = np.square(self.V[pos_items[idx], :]).sum() + \
                np.square(self.V[neg_items[idx], :]).sum() + \
                np.square(self.W[user_items_t, :]).sum() + \
                np.square(self.b[pos_items[idx]]) + \
                np.square(self.eta).sum() + \
                np.square(self.etaU[user_id, :]).sum()

            self.loss += - \
                np.log(1 / (1 + np.exp(-np.clip((R_pos - R_neg), -10, 10)))
                    ) + 0.5 * self.alpha * Reg

            # Compute gradients of each parameters that involved.
            delta = 1 / (1 + np.exp(-np.clip((R_neg - R_pos), -10, 10)))
            gradb_pos = self.alpha * self.b[pos_items[idx]] - delta
            gradb_neg = self.alpha * self.b[neg_items[idx]] + delta
            gradV_pos = self.alpha * self.V[pos_items[idx], :] - delta * U
            gradV_neg = self.alpha * self.V[neg_items[idx], :] + delta * U
            gradeta = self.alpha * self.eta - \
                delta * np.matmul(
                    self.W[user_items[:-self.L-1:-1], :],
                    self.V[pos_items[idx], :] - self.V[neg_items[idx], :],
                )
            gradetaU = self.alpha * self.etaU[user_id, :] - \
                delta * np.matmul(
                    self.W[user_items[:-self.L-1:-1], :],
                    self.V[pos_items[idx], :] - self.V[neg_items[idx], :],
            )
            gradW_longterm = self.alpha * self.W[user_items[:-self.L], :] - \
                delta * (self.V[pos_items[idx]] / np.sqrt(t - 1) -
                        self.V[neg_items[idx]] / np.sqrt(t))
            gradW_shortterm = self.alpha * self.W[user_items[:-self.L-1:-1], :] - \
                delta * np.outer(
                    self.eta + self.etaU[user_id, :],
                    self.V[pos_items[idx], :] - self.V[neg_items[idx], :],
            ) - \
                delta * (self.V[pos_items[idx]] / np.sqrt(t - 1) -
                        self.V[neg_items[idx]] / np.sqrt(t))
            gradW_neg = self.alpha * self.W[neg_items[idx], :] - \
                delta * (-self.V[neg_items[idx]] / np.sqrt(t))

            # Update parameters.
            self.b[pos_items[idx]] -= self.gamma * gradb_pos
            self.b[neg_items[idx]] -= self.gamma * gradb_neg
            self.V[pos_items[idx], :] -= self.gamma * gradV_pos
            self.V[neg_items[idx], :] -= self.gamma * gradV_neg
            self.eta -= self.gamma * gradeta
            self.etaU -= self.gamma * gradetaU
            self.W[user_items[:-self.L], :] -= self.gamma * gradW_longterm
            self.W[user_items[:-self.L-1:-1],
                :] -= self.gamma * gradW_shortterm
            self.W[neg_items[idx], :] -= self.gamma * gradW_neg

    def eval(self, train_data: np.ndarray, valid_data: np.ndarray, nega_data: np.ndarray) -> float:

        print("Loss = {:.8} | ".format(self.loss), end="")
        self.loss = 0.

        valid_data = np.column_stack([valid_data, nega_data])
        T = (train_data != -1).sum(axis=1)
        Recall_at_K = 0
        for user_id in range(self.UserNum):
            t = T[user_id]
            user_items = train_data[user_id, :t]
            R = np.zeros([valid_data.shape[1]])

            longterm = self.W[user_items, :].sum(axis=0) / np.sqrt(t)
            shortterm = np.matmul(
                self.eta + self.etaU[user_id, :], self.W[user_items[:-self.L-1:-1], :])
            U = longterm + shortterm

            R = self.b[valid_data[user_id, :]] + \
                np.matmul(self.V[valid_data[user_id, :], :], U)
            topK = np.argpartition(R, -self.K)[-self.K:]
            if 0 in topK:
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
