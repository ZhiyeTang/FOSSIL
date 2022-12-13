import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_variable(shape: list, *params, initializer="xavier",):

    out = torch.empty(shape)
    if initializer == "xavier":
        if not params:
            out = nn.init.xavier_uniform_(out, params[0])
        else:
            out = nn.init.xavier_uniform_(out)
    elif initializer == "trunc_norm":
        out = nn.init.trunc_normal_(out, params[0], params[1])

    return out


class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_heads: int,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(SelfAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.K = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.V = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.softmax = nn.Softmax().to(device)
        self.dropout_1 = nn.Dropout(self.dropout_rate).to(device)
        self.dropout_2 = nn.Dropout(self.dropout_rate).to(device)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ):

        Qx = self.Q @ x
        Kx = self.K @ x
        Vx = self.V @ x
        Qh = torch.concat(torch.split(Qx, self.num_heads, dim=2), dim=0)
        Kh = torch.concat(torch.split(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.split(Vx, self.num_heads, dim=2), dim=0)

        outputs = torch.matmul(Qh, Kh.transpose(1, 2)) / (Kh.shape[-1] ** 0.5)
        tril = torch.tril(torch.ones_like(outputs[0, :, :]))
        casuality_mask = torch.tile(torch.unsqueeze(tril, 0), [
                                    outputs.shape[0], 1, 1])
        outputs = torch.where(torch.eq(casuality_mask, 0),
                              torch.ones_like(outputs)*(-2**32+1), outputs)

        key_mask = torch.tile(
            padding_mask, [self.num_heads, 1, self.Q.shape[1]]).transpose(1, 2)
        outputs = torch.where(torch.eq(key_mask, 0),
                              torch.ones_like(outputs)*(-2**32+1), outputs)

        outputs = self.softmax(outputs)

        query_mask = torch.tile(padding_mask, [self.num_heads, 1, x.shape[1]])
        outputs *= query_mask

        outputs = self.dropout_1(outputs)
        attention = torch.mean(torch.stack(torch.split(
            outputs[:, -1], self.num_heads, dim=0), dim=0), dim=0)

        outputs = torch.matmul(outputs, Vh)
        outputs = torch.concat(torch.split(
            outputs, self.num_heads, dim=0), dim=-1)
        outputs = self.dropout_2(outputs)
        outputs += x

        return outputs, attention


class LocationBasedAttentionBlock(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_heads: int,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(LocationBasedAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q = nn.Parameter(
            get_variable([1, num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.K = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.V = nn.Parameter(
            get_variable([num_inputs, num_units]),
            requires_grad=True,
        ).to(device)
        self.softmax = nn.Softmax().to(device)
        self.dropout_1 = nn.Dropout(self.dropout_rate).to(device)
        self.dropout_2 = nn.Dropout(self.dropout_rate).to(device)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ):

        Qx = self.Q @ x
        Kx = self.K @ x
        Vx = self.V @ x
        Qh = torch.concat(torch.split(Qx, self.num_heads, dim=2), dim=0)
        Kh = torch.concat(torch.split(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.split(Vx, self.num_heads, dim=2), dim=0)

        outputs = torch.matmul(Qh, Kh.transpose(1, 2))

        key_mask = torch.tile(
            padding_mask, [self.num_heads, 1, self.Q.shape[1]]).transpose(1, 2)
        outputs = torch.where(torch.eq(key_mask, 0),
                              torch.ones_like(outputs)*(-2**32+1), outputs)

        outputs = self.softmax(outputs)

        outputs = self.dropout_1(outputs)
        attention = torch.split(outputs[:, -1], self.num_heads, dim=0)

        outputs = torch.matmul(outputs, Vh)
        outputs = torch.concat(torch.split(
            outputs, self.num_heads, dim=0), dim=-1)
        outputs = self.dropout_2(outputs)

        return outputs, attention


class FeedForward(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(FeedForward, self).__init__()

        self.conv_1 = nn.Conv1d(num_inputs, num_units, 1).to(device)
        self.dropout_1 = nn.Dropout(p=dropout_rate).to(device)
        self.conv_2 = nn.Conv1d(num_units, num_inputs, 1).to(device)
        self.dropout_2 = nn.Dropout(p=dropout_rate).to(device)

    def forward(self, x: torch.Tensor):

        outputs = F.relu(self.conv_1(x))
        outputs = self.dropout_1(outputs)
        outputs = self.conv_2(outputs)
        outputs = self.dropout_2(outputs)
        outputs += x

        return outputs


class ItemSimilarityGatingBlock(nn.Module):

    def __init__(
        self,
        num_units: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(ItemSimilarityGatingBlock, self).__init__()

        self.num_units = num_units

        self.dropout_1 = nn.Dropout(p=dropout_rate).to(device)
        self.dropout_2 = nn.Dropout(p=dropout_rate).to(device)

    def forward(
        self,
        embeds_1: torch.Tensor,
        embeds_2: torch.Tensor,
        embeds_3: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):

        inputs = torch.concat(
            [embeds_1, embeds_2, embeds_3], -1).reshape([-1, self.num_units*3])
        inputs = self.dropout_1(inputs)

        logits = torch.matmul(inputs, weight) + bias
        logits = self.dropout_2(logits)
        logits = logits.reshape([-1, embeds_2.shape[1], 1])

        outputs = F.sigmoid(logits)

        return outputs


class FissaNetwork(nn.Module):

    def __init__(
        self,
        num_items: int,
        max_len: int,
        num_units: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super(FissaNetwork).__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.item_embedding = nn.Parameter(
            get_variable([self.num_items, self.num_units]),
            requires_grad=True,
        ).to(device)
        self.pos_embedding = nn.Parameter(
            get_variable([self.max_len, self.num_units]),
            requires_grad=True,
        ).to(device)
        self.inputs_dropout = nn.Dropout(p=dropout_rate)
        self.inputs_layer_norm = nn.LayerNorm(num_units, eps=1e-8)

        self.SABs = [
            {
                "SAB": SelfAttentionBlock(
                    num_inputs=self.num_units,
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout_rate=dropout_rate,
                    device=device,
                ),
                "LN1": nn.LayerNorm(
                    num_units,
                    eps=1e-8,
                ),
                "FF": FeedForward(
                    num_inputs=self.num_units,
                    num_units=self.num_units,
                    dropout_rate=dropout_rate,
                    device=device
                ),
                "LN2": nn.LayerNorm(
                    num_units,
                    eps=1e-8,
                ),
            }
        ] * self.num_blocks
        self.local_dropout = nn.Dropout(p=dropout_rate)

        self.LBAB = {
            "LBAB": LocationBasedAttentionBlock(
                num_inputs=self.num_units,
                num_heads=self.num_heads,
                num_units=self.num_units,
                dropout_rate=dropout_rate,
                device=device
            ),
            "LN1": nn.LayerNorm(
                num_units,
                eps=1e-8,
            ),
            "FF": FeedForward(
                num_inputs=self.num_units,
                num_units=self.num_units,
                dropout_rate=dropout_rate,
                device=device
            ),
            "LN2": nn.LayerNorm(
                num_units,
                eps=1e-8,
            ),
        }
        self.global_dropout = nn.Dropout(p=dropout_rate)

        self.gated_weight = nn.Parameter(
            get_variable(
                [self.num_units*3, 1],
                [0.0, (2 / (3 * self.num_units + 1)) ** 0.5],
                initializer="trunc_norm",
            ),
            requires_grad=True,
        ).to(device)
        self.gated_bias = nn.Parameter(
            get_variable(
                [1, 1],
                [0.0, (2 / (2 * self.num_units + 1)) ** 0.5],
                initializer="trunc_norm",
            ),
            requires_grad=True,
        ).to(device)
        self.ISGB = ItemSimilarityGatingBlock(
            self.num_units, dropout_rate, device)

        self.outputs_layer_norm = nn.LayerNorm(num_units, eps=1e-8)

    def forward(
        self,
        user_items: torch.Tensor,
        padding_mask: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):

        item_embeds = self.item_embedding.index_select(0, user_items)
        pos_embeds = self.pos_embedding.index_select(
            0, torch.arange(self.max_len))

        inputs = item_embeds + pos_embeds
        inputs = self.inputs_dropout(inputs)
        inputs *= padding_mask
        inputs = self.inputs_layer_norm(inputs)

        # local representation
        loc_embeds = inputs
        for _, block in enumerate(self.SABs):

            loc_embeds, _ = block["SAB"].forward(loc_embeds, padding_mask)
            loc_embeds = block["LN1"].forward(loc_embeds)

            loc_embeds = block["FF"].forward(loc_embeds)
            loc_embeds *= padding_mask
            loc_embeds = block["LN2"].forward(loc_embeds)

        # global representation
        glo_embeds = self.LBAB["LBAB"].forward(item_embeds, padding_mask)
        glo_embeds = self.LBAB["LN1"].forward(glo_embeds)
        glo_embeds = self.LBAB["FF"].forward(glo_embeds)
        glo_embeds = self.LBAB["LN2"].forward(glo_embeds)

        # hybrid representation
        positive_embeds = item_embeds.index_select(0, pos_items)
        negative_embeds = item_embeds.index_select(0, neg_items)

        gated_value = self.ISGB.forward(
            torch.tile(glo_embeds, [2, self.max_len, 1]),
            torch.tile(item_embeds, [2, 1, 1]),
            torch.concat([positive_embeds, negative_embeds], 0),
            self.gated_weight,
            self.gated_bias,
        )
        gated_value = F.sigmoid(gated_value)

        tiled_loc_embeds = torch.tile(loc_embeds, [2, 1, 1])
        tiled_glo_embeds = torch.tile(glo_embeds, [2, 1, 1])

        outputs = tiled_loc_embeds * gated_value + \
            tiled_glo_embeds * (1 - gated_value)
        outputs *= torch.tile(padding_mask, [2, 1, 1])
        outputs = self.outputs_layer_norm(outputs)

        outputs_positive = outputs[:user_items.shape[0]]
        outputs_negative = outputs[user_items.shape[0]:]

        outputs = torch.concat(
            [
                torch.sum(outputs_positive*positive_embeds, -1),
                torch.sum(outputs_negative*negative_embeds, -1),
            ],
            dim=0,
        )

        return outputs

    def inference(
        self,
        user_items: torch.Tensor,
        padding_mask: torch.Tensor,
        candidate_items: torch.Tensor,
    ):

        item_embeds = self.item_embedding.index_select(0, user_items)
        candidate_embeds = self.pos_embedding.index_select(0, candidate_items)
        pos_embeds = self.pos_embedding.index_select(
            0, torch.arange(self.max_len))

        inputs = item_embeds + pos_embeds
        inputs = self.inputs_dropout(inputs)
        inputs *= padding_mask
        inputs = self.inputs_layer_norm(inputs)

        # local representation
        loc_embeds = inputs
        for _, block in enumerate(self.SABs):

            loc_embeds, _ = block["SAB"].forward(loc_embeds, padding_mask)
            loc_embeds = block["LN1"].forward(loc_embeds)

            loc_embeds = block["FF"].forward(loc_embeds)
            loc_embeds *= padding_mask
            loc_embeds = block["LN2"].forward(loc_embeds)

        # global representation
        glo_embeds = self.LBAB["LBAB"].forward(item_embeds, padding_mask)
        glo_embeds = self.LBAB["LN1"].forward(glo_embeds)
        glo_embeds = self.LBAB["FF"].forward(glo_embeds)
        glo_embeds = self.LBAB["LN2"].forward(glo_embeds)

        # hybrid representation
        gated_value = self.ISGB.forward(
            torch.tile(glo_embeds, [1, 101, 1]),
            torch.tile(item_embeds[:, -1, :].unsqueeze(1), [1, 101, 1]),
            candidate_embeds,
            self.gated_weight,
            self.gated_bias,
        )
        gated_value = F.sigmoid(gated_value)

        tiled_loc_embeds = torch.tile(
            loc_embeds[:, -1, :].unsqueeze(1), [1, 101, 1])
        tiled_glo_embeds = torch.tile(glo_embeds, [1, 101, 1])

        outputs = tiled_loc_embeds * gated_value + \
            tiled_glo_embeds * (1 - gated_value)
        outputs = self.outputs_layer_norm(outputs)

        return outputs


class Algorithm:

    def __init__(self, config):

        self.UserNum = config["UserNum"]
        self.ItemNum = config["ItemNum"]
        self.MaxLen = config["MaxLen"]

        self.DropoutRate = config["DropoutRate"]
        self.UnitNum = config["UnitNum"]
        self.BlockNum = config["BlockNum"]
        self.HeadNum = config["HeadNum"]

        self.device = config["device"]
        self.Network = FissaNetwork(
            num_items=self.ItemNum,
            max_len=self.MaxLen,
            num_units=self.UnitNum,
            num_heads=self.HeadNum,
            num_blocks=self.BlockNum,
            dropout_rate=self.DropoutRate,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            self.Network.parameters(),
            lr=config["lr"],
        )

        self.L = config["L"]
        self.K = config["K"]

    def train(self, user_id: int, user_items, pos_items, neg_items) -> None:

        user_items_filled = -np.ones([self.MaxLen])
        user_items_filled[:len(user_items)] = user_items
        user_items = np.tile(user_items_filled, [self.MaxLen, 1])
        padding_mask = np.zeros_like(user_items)
        padding_mask[user_items!=-1] = 1

        user_items = torch.as_tensor(user_items).to(self.device)
        padding_mask = torch.as_tensor(padding_mask).to(self.device)
        pos_items = torch.as_tensor(pos_items).to(self.device)
        neg_items = torch.as_tensor(neg_items).to(self.device)

        outputs = self.Network.forward(
            user_items.unsqueeze(0),
            padding_mask.unsqueeze(0),
            pos_items.unsqueeze(0),
            neg_items.unsqueeze(0),
        )
        print(outputs.shape)

        loss = torch.mean(
            -torch.log(F.sigmoid(outputs[0]) + 1e-24)\
            -torch.log(1 - F.sigmoid(outputs[1]) + 1e-24)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self, train_data, valid_data, nega_data) -> float:

        user_items = np.tile(np.expand_dims(train_data, -1), [1, 1, self.MaxLen])
        padding_mask = np.zeros_like(user_items)
        padding_mask[user_items!=-1] = 1
        candidate_items = np.column_stack([valid_data, nega_data])

        user_items = torch.as_tensor(user_items).to(self.device)
        padding_mask = torch.as_tensor(padding_mask).to(self.device)
        candidate_items = torch.as_tensor(candidate_items).to(self.device)

        Recall_at_K = 0.
        for user_id in range(self.UserNum):
            outputs = self.Network.inference(
                user_items[user_id, :, :],
                padding_mask[user_id, :, :],
                candidate_items[user_id, :],
            )

            R = outputs.detach_().to("cpu").numpy()
            topK = np.argpartition(R, -self.K)[-self.K:]
            if 0 in topK:
                Recall_at_K += 1
        
        return Recall_at_K

