import os
from random import shuffle
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_utils.raw_dataset_loader import RawDataLoader
from data_utils.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
    Dataset,
)
from utils.models_setup import generate_model
from utils.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt


def train_validate_test_normal(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config,
    model_with_config_name,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        """
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        """
    model.to(device)
    num_epoch = config["num_epoch"]
    trainlib = []
    vallib = []
    testlib = []  # total loss tracking for train/vali/test
    tasklib = []
    tasklib_test = []
    tasklib_vali = []  # loss tracking for summation across all atoms/nodes
    tasklib_nodes = []
    tasklib_test_nodes = []
    tasklib_vali_nodes = []  # probably not needed

    x_atomfeature = []
    for data in test_loader.dataset:
        x_atomfeature.append(data.x)
    if 0:  # visualizing of initial conditions
        test_rmse = test(test_loader, model, config["output_dim"])
        true_values = test_rmse[3]
        predicted_values = test_rmse[4]
        for ihead in range(model.num_heads):
            visualizer = Visualizer(model_with_config_name)
            visualizer.add_test_values(
                true_values=true_values[ihead], predicted_values=predicted_values[ihead]
            )
            visualizer.create_scatter_plot_atoms_hist(ihead, x_atomfeature, -1)

    for epoch in range(0, num_epoch):
        train_mae, train_taskserr, train_taskserr_nodes = train(
            train_loader, model, optimizer, config["output_dim"]
        )
        val_mae, val_taskserr, val_taskserr_nodes = validate(
            val_loader, model, config["output_dim"]
        )
        test_rmse = test(test_loader, model, config["output_dim"])
        scheduler.step(val_mae)
        writer.add_scalar("train error", train_mae, epoch)
        writer.add_scalar("validate error", val_mae, epoch)
        writer.add_scalar("test error", test_rmse[0], epoch)
        for ivar in range(model.num_heads):
            writer.add_scalar(
                "train error of task" + str(ivar), train_taskserr[ivar], epoch
            )

        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}"
        )
        print("Tasks MAE:", train_taskserr)

        trainlib.append(train_mae)
        vallib.append(val_mae)
        testlib.append(test_rmse[0])
        tasklib.append(train_taskserr)
        tasklib_vali.append(val_taskserr)
        tasklib_test.append(test_rmse[1])

        tasklib_nodes.append(train_taskserr_nodes)
        tasklib_vali_nodes.append(val_taskserr_nodes)
        tasklib_test_nodes.append(test_rmse[2])

        ###tracking the solution evolving with training
        # true_values=test_rmse[3]; predicted_values=test_rmse[4]
        # for ihead in range(model.num_heads):
        #    visualizer = Visualizer(model_with_config_name)
        #    visualizer.add_test_values(
        #        true_values=true_values[ihead], predicted_values=predicted_values[ihead]
        #    )
        #    visualizer.create_scatter_plot_atoms_hist(ihead, x_atomfeature,epoch)

    # At the end of training phase, do the one test run for visualizer to get latest predictions
    test_rmse, test_taskserr, test_taskserr_nodes, true_values, predicted_values = test(
        test_loader, model, config["output_dim"]
    )

    for ihead in range(model.num_heads):
        visualizer = Visualizer(model_with_config_name)
        visualizer.add_test_values(
            true_values=true_values[ihead], predicted_values=predicted_values[ihead]
        )
        visualizer.create_scatter_plot(ihead)
        visualizer.create_scatter_plot_atoms(ihead, x_atomfeature)

    ######plot loss history#####
    visualizer.plot_history(
        trainlib,
        vallib,
        testlib,
        tasklib,
        tasklib_vali,
        tasklib_test,
        tasklib_nodes,
        tasklib_vali_nodes,
        tasklib_test_nodes,
        model.hweights,
    )


def train(loader, model, opt, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)

    model.train()
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        loss, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        loss.backward()
        opt.step()
        total_error += loss.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs
    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def validate(loader, model, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
    )


@torch.no_grad()
def test(loader, model, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    tasks_error = np.zeros(model.num_heads)
    tasks_noderr = np.zeros(model.num_heads)
    model.eval()
    true_values = [[] for _ in range(model.num_heads)]
    predicted_values = [[] for _ in range(model.num_heads)]
    IImean = [i for i in range(sum(model.head_dims))]
    if model.inllloss == 1:
        IImean = [i for i in range(sum(model.head_dims) + model.num_heads)]
        [
            IImean.remove(sum(model.head_dims[: ihead + 1]) + (ihead + 1) * 1 - 1)
            for ihead in range(model.num_heads)
        ]
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        error, tasks_rmse, tasks_nodes = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs
        for itask in range(len(tasks_rmse)):
            tasks_error[itask] += tasks_rmse[itask].item() * data.num_graphs
            tasks_noderr[itask] += tasks_nodes[itask].item() * data.num_graphs

        ytrue = torch.reshape(data.y, (-1, sum(model.head_dims)))
        for ihead in range(model.num_heads):
            isum = sum(model.head_dims[: ihead + 1])
            true_values[ihead].extend(
                ytrue[:, isum - model.head_dims[ihead] : isum].tolist()
            )
            predicted_values[ihead].extend(
                pred[:, IImean[isum - model.head_dims[ihead] : isum]].tolist()
            )

    return (
        total_error / len(loader.dataset),
        tasks_error / len(loader.dataset),
        tasks_noderr / len(loader.dataset),
        true_values,
        predicted_values,
    )


def dataset_loading_and_splitting(
    config: {},
    chosen_dataset_option: Dataset,
):

    if chosen_dataset_option == Dataset.CuAu:
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        return split_dataset(
            dataset=dataset_CuAu,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    elif chosen_dataset_option == Dataset.FePt:
        dataset_FePt = load_data(Dataset.FePt.value, config)
        return split_dataset(
            dataset=dataset_FePt,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    elif chosen_dataset_option == Dataset.FeSi:
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        return split_dataset(
            dataset=dataset_FeSi,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    else:
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        dataset_FePt = load_data(Dataset.FePt.value, config)
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        if chosen_dataset_option == Dataset.CuAu_FePt_SHUFFLE:
            dataset_CuAu.extend(dataset_FePt)
            dataset_combined = dataset_CuAu
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:

            return combine_and_split_datasets(
                dataset1=dataset_CuAu,
                dataset2=dataset_FePt,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
            return combine_and_split_datasets(
                dataset1=dataset_FePt,
                dataset2=dataset_CuAu,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_FeSi_SHUFFLE:
            dataset_FePt.extend(dataset_FeSi)
            dataset_combined = dataset_FePt
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )


def split_dataset(dataset: [], batch_size: int, perc_train: float):
    perc_val = (1 - perc_train) / 2
    data_size = len(dataset)
    train_loader = DataLoader(
        dataset[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[int(data_size * perc_train) : int(data_size * (perc_train + perc_val))],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset[int(data_size * (perc_train + perc_val)) :],
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [], dataset2: [], batch_size: int, perc_train: float
):
    data_size = len(dataset1)
    train_loader = DataLoader(
        dataset1[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset1[int(data_size * perc_train) :],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_data(dataset_option, config):
    transform_raw_data_to_serialized()
    files_dir = (
        f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
    )

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset = loader.load_serialized_data(
        dataset_path=files_dir,
        config=config,
    )

    return dataset


def transform_raw_data_to_serialized():
    # Loading raw data if necessary
    raw_datasets = ["CuAu_32atoms", "FePt_32atoms", "FeSi_1024atoms"]
    if len(
        os.listdir(os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset")
    ) < len(raw_datasets):
        for raw_dataset in raw_datasets:
            files_dir = (
                os.environ["SERIALIZED_DATA_PATH"]
                + "/dataset/"
                + raw_dataset
                + "/output_files/"
            )
            loader = RawDataLoader()
            loader.load_raw_data(dataset_path=files_dir)
