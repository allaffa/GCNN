import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch.nn import GaussianNLLLoss
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
import sys


class PNNStack(torch.nn.Module):
    def __init__(
        self,
        deg,
        input_dim,
        output_dim,
        num_nodes,
        hidden_dim,
        num_conv_layers,
        num_shared=1,
        ihpwloss=1,  # if =1, considering weighted losses for different tasks and treat the weights as hyper parameters
        hweights=[1.0, 1.0, 1.0],  # weights for losses of different tasks
        inllloss=0,  # if =1, using the scalar uncertainty as weights, as in paper
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    ):
        super(PNNStack, self).__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]

        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            PNAConv(
                in_channels=input_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        for _ in range(self.num_conv_layers):
            conv = PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))
        ############multiple heads/taks################
        denselayers = []  # shared dense layers, before mutli-heads
        for ishare in range(num_shared):
            denselayers.append(Linear(self.hidden_dim, self.hidden_dim))
            denselayers.append(ReLU())
        self.shared = Sequential(*denselayers)

        # currently, only two types of outputs are considered, graph-level scalars and nodes-level vectors with num_nodes dimension, or mixed or the two
        if output_dim < num_nodes:  # all graph-level outputs
            self.num_heads = output_dim
            outputs_dims = [1 for _ in range(self.num_heads)]
        elif output_dim % num_nodes == 0:  # all node-level outputs
            self.num_heads = output_dim // num_nodes
            outputs_dims = [num_nodes for _ in range(self.num_heads)]
        else:  # mixed graph-level and node-level
            self.num_heads = output_dim % num_nodes + output_dim // num_nodes
            outputs_dims = [
                1 if ihead < output_dim % num_nodes else num_nodes
                for ihead in range(self.num_heads)
            ]

        self.num_heads = len(outputs_dims)  # number of heads/tasks
        self.head_dims = outputs_dims
        self.heads = ModuleList()
        self.inllloss = inllloss
        self.ihpwloss = ihpwloss
        if self.ihpwloss * self.inllloss == 1:
            sys.exit("Error: both ihpwloss and inllloss are set to 1.")
        if self.ihpwloss == 1:
            if len(hweights) != self.num_heads:
                sys.exit(
                    "Inconsistent number of loss weights and tasks: "
                    + str(len(hweights))
                    + " VS "
                    + str(self.num_heads)
                )
                # print('Inconsistent number of loss weights and tasks: '+str(len(hweights)) + ' VS ' +str(self.num_heads))
                # self.hweights = [4**ihead for ihead in range(self.num_heads)]
            else:
                self.hweights = hweights
            weightabssum = sum(abs(number) for number in self.hweights)
            self.hweights = [iw / weightabssum for iw in self.hweights]
        for ihead in range(self.num_heads):
            mlp = Sequential(
                Linear(self.hidden_dim, 50),
                ReLU(),
                Linear(50, 25),
                ReLU(),
                Linear(
                    25, self.head_dims[ihead] + inllloss * 1
                ),  # for log(noise or uncertainty output)
            )
            self.heads.append(mlp)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        ### encoder part ####
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))
        x = global_mean_pool(x, batch)
        x = self.shared(x)  # shared dense layers
        #### multi-head decoder part####
        outputs = []
        for headloc in self.heads:
            outputs.append(headloc(x))
        return torch.cat(outputs, dim=1)

    def loss(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return F.l1_loss(pred, value)

    def loss_rmse(self, pred, value):

        if self.inllloss == 1:
            return self.loss_NLL(pred, value)
        elif self.ihpwloss == 1:
            return self.loss_hpweighted(pred, value)

        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return torch.sqrt(F.mse_loss(pred, value)), [], []

    def loss_NLL(self, pred, value):  # negative log likelihood loss
        # uncertainty to weigh losses in https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape[0] != value_shape[0]:
            value = torch.reshape(value, (pred_shape[0], -1))
            if value.shape[1] != pred.shape[1] - self.num_heads:
                sys.exit(
                    "expected feature dims="
                    + str(pred.shape[1] - self.num_heads)
                    + "; actual="
                    + str(value.shape[1])
                )

        nll_loss = 0
        tasks_rmseloss = []
        loss = GaussianNLLLoss()
        for ihead in range(self.num_heads):
            isum = sum(self.head_dims[: ihead + 1])
            ivar = isum + (ihead + 1) * 1 - 1

            head_var = torch.exp(pred[:, ivar])
            head_pre = pred[:, ivar - self.head_dims[ihead] : ivar]
            head_val = value[:, isum - self.head_dims[ihead] : isum]
            nll_loss += loss(head_pre, head_val, head_var)
            tasks_rmseloss.append(torch.sqrt(F.mse_loss(head_pre, head_val)))

        return nll_loss, tasks_rmseloss, []

    def loss_hpweighted(
        self, pred, value
    ):  # weights for difficult tasks as hyper-parameters
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)

        tot_loss = 0
        tasks_rmseloss = []
        tasks_nodes = []
        for ihead in range(self.num_heads):
            isum = sum(self.head_dims[: ihead + 1])

            head_pre = pred[:, isum - self.head_dims[ihead] : isum]
            head_val = value[:, isum - self.head_dims[ihead] : isum]

            tot_loss += (
                torch.sqrt(F.mse_loss(head_pre, head_val)) * self.hweights[ihead]
            )
            tasks_nodes.append(torch.sqrt(F.mse_loss(head_pre, head_val)))
            # loss of summation across nodes/atoms
            tasks_rmseloss.append(
                torch.sqrt(F.mse_loss(torch.sum(head_pre, 1), torch.sum(head_val, 1)))
            )

        return tot_loss, tasks_rmseloss, tasks_nodes

    def __str__(self):
        return "PNNStack"
