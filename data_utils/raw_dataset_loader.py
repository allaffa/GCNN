import os
import numpy as np
import pickle
import pathlib
from abc import ABCMeta, abstractmethod

import torch
from torch_geometric.data import Data
from torch import tensor

from data_utils.helper_functions import tensor_divide


def element_name_to_proton_number(element: str):

    if element == "H":
        return 1
    elif element == "C":
        return 6
    elif element == "N":
        return 7
    elif element == "O":
        return 8
    elif element == "F":
        return 9
    else:
        print(element, ": Sorry Max & Pei")
        exit()


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data(dataset_path: str)
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    __metaclass__ = ABCMeta

    node_position_dim = []
    node_position_col = []
    node_feature_dim = []
    node_feature_col = []
    graph_feature_dim = []
    graph_feature_col = []
    num_nodes = 0
    format = ""

    def __init__(self, dataset_path: str, config):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.

        Parameters
        ----------
        dataset_path: str
            Directory path where raw files are stored.
        config: shows the target variables information, e.g, location and dimension, in data file
        """

        self.node_position_dim = config["node_positions"]["dim"]
        self.node_position_col = config["node_positions"]["column_index"]
        self.node_feature_dim = config["node_features"]["dim"]
        self.node_feature_col = config["node_features"]["column_index"]
        self.graph_feature_dim = config["graph_features"]["dim"]
        self.graph_feature_col = config["graph_features"]["column_index"]
        self.num_nodes = int(config["num_nodes"])
        self.format = config["format"]

        dataset = []
        assert (
            len(os.listdir(dataset_path)) > 0
        ), "No data files provided in {}!".format(dataset_path)

        for filename in os.listdir(dataset_path):
            if filename == ".DS_Store":
                continue
            f = open(os.path.join(dataset_path, filename), "r", encoding="utf-8")
            all_lines = f.readlines()
            data_object = self._extract_dataset(lines=all_lines)
            dataset.append(data_object)
            f.close()

        if self.format == "LSMS":
            for idx, data_object in enumerate(dataset):
                dataset[idx] = self._update_dataset(data_object)

        (
            dataset_normalized,
            minmax_node_feature,
            minmax_graph_feature,
        ) = self.__normalize_dataset(dataset=dataset)

        serial_data_name = config["name"]
        serial_data_path = (
            os.environ["SERIALIZED_DATA_PATH"]
            + "/serialized_dataset/"
            + serial_data_name
            + ".pkl"
        )

        with open(serial_data_path, "wb") as f:
            pickle.dump(minmax_node_feature, f)
            pickle.dump(minmax_graph_feature, f)
            pickle.dump(dataset_normalized, f)

    @abstractmethod
    def _extract_dataset(self, lines: [str]):
        """Transforms raw data to torch_geometric Data and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        raise NotImplementedError("Must override _extract_dataset")

    def _extract_graph_features(self, line: [str]):
        graph_feat = []
        for item in range(len(self.graph_feature_dim)):
            for icomp in range(self.graph_feature_dim[item]):
                it_comp = self.graph_feature_col[item] + icomp
                graph_feat.append(float(line[it_comp].strip()))
        return tensor(graph_feat)

    def _extract_node_features(self, lines: [str]):
        node_position_matrix = []
        node_feature_matrix = []
        for line in lines:
            node_feat = line.split()
            pos = []
            for d in range(self.node_position_dim):
                pos.append(float(node_feat[self.node_position_col[d]].strip()))
            node_position_matrix.append(pos)

            node_feature = []
            for item in range(len(self.node_feature_dim)):
                for icomp in range(self.node_feature_dim[item]):
                    it_comp = self.node_feature_col[item] + icomp
                    try:
                        comp = float(node_feat[it_comp].strip())
                    except ValueError:
                        comp = float(
                            element_name_to_proton_number(node_feat[it_comp].strip())
                        )
                    node_feature.append(comp)

            node_feature_matrix.append(node_feature)

        return tensor(node_position_matrix), tensor(node_feature_matrix)

    def _update_dataset(self, data_object: Data):
        """Update dataset before normalization.
        Parameters
        ----------
        data_object: Data
            Data object representing structure of a graph sample.

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        return data_object

    def __normalize_dataset(self, dataset: [Data]):
        """Performs the normalization on Data objects and returns the normalized dataset.

        Parameters
        ----------
        dataset: [Data]
            List of Data objects representing structures of graphs.

        Returns
        ----------
        [Data]
            Normalized dataset.
        """
        num_of_nodes = len(dataset[0].x)
        num_node_features = dataset[0].x.shape[1]
        num_graph_features = len(dataset[0].y)

        minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        minmax_node_feature = np.full((2, num_of_nodes, num_node_features), np.inf)
        minmax_graph_feature[1, :] *= -1
        minmax_node_feature[1, :, :] *= -1

        for data in dataset:
            # find maximum and minimum values for graph level features
            for ifeat in range(num_graph_features):
                minmax_graph_feature[0, ifeat] = min(
                    data.y[ifeat], minmax_graph_feature[0, ifeat]
                )
                minmax_graph_feature[1, ifeat] = max(
                    data.y[ifeat], minmax_graph_feature[1, ifeat]
                )
            # find maximum and minimum values for node level features
            for ifeat in range(num_node_features):
                minmax_node_feature[0, :, ifeat] = np.minimum(
                    data.x[:, ifeat].numpy(), minmax_node_feature[0, :, ifeat]
                )
                minmax_node_feature[1, :, ifeat] = np.maximum(
                    data.x[:, ifeat].numpy(), minmax_node_feature[1, :, ifeat]
                )

        for data in dataset:
            for ifeat in range(num_graph_features):
                data.y[ifeat] = tensor_divide(
                    (data.y[ifeat] - minmax_graph_feature[0, ifeat]),
                    (minmax_graph_feature[1, ifeat] - minmax_graph_feature[0, ifeat]),
                )
            for ifeat in range(num_node_features):
                data.x[:, ifeat] = tensor_divide(
                    (data.x[:, ifeat] - minmax_node_feature[0, :, ifeat]),
                    (
                        minmax_node_feature[1, :, ifeat]
                        - minmax_node_feature[0, :, ifeat]
                    ),
                )

        return dataset, minmax_node_feature, minmax_graph_feature


class LSMSDataLoader(RawDataLoader):
    def __init__(self, dataset_path: str, config):
        super().__init__(dataset_path, config)

    def _extract_dataset(self, lines: [str]):
        data_object = Data()

        graph_feat = lines[0].split()
        data_object.y = self._extract_graph_features(graph_feat)

        node_lines = lines[1 : self.num_nodes + 1]
        data_object.pos, data_object.x = self._extract_node_features(node_lines)

        return data_object

    def _update_dataset(self, data_object: Data):
        """Update charge density."""
        num_of_protons = data_object.x[0]
        charge_density = data_object.x[1]
        charge_density -= num_of_protons
        data_object.x[1] = charge_density
        return data_object


class XYZDataLoader(RawDataLoader):
    def __init__(self, dataset_path: str, config):
        super().__init__(dataset_path, config)

    def _extract_dataset(self, lines: [str]):
        data_object = Data()

        graph_feat = lines[1].split()
        data_object.y = self._extract_graph_features(graph_feat)

        node_lines = lines[2 : self.num_nodes + 2]
        data_object.pos, data_object.x = self._extract_node_features(node_lines)

        return data_object
