import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch_geometric.data import Data

from data_utils.dataset_descriptors import AtomFeatures
from data_utils.helper_functions import (
    distance_3D,
    remove_collinear_candidates,
    order_candidates,
    resolve_neighbour_conflicts,
)


class SerializedDataLoader:
    """A class used for loading existing structures from files that are lists of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_serialized_data method.

    Methods
    -------
    load_serialized_data(dataset_path: str, config: dict)
        Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.
    """

    config = {}
    radius = 0.0
    periodic = False
    num_local = 0

    def __init__(self, config):
        """
        Parameters
        ----------
        config: dict
            Dictionary containing information needed to load the data and transform it, respectively: atom_features, radius, max_num_node_neighbours and predicted_value_option.
        """
        # This is only the NeuralNetwork section of the config.
        self.config = config
        self.radius = config["Architecture"]["radius"]
        self.periodic = config["Architecture"]["periodic"] == "True"

    def load_serialized_data(self, dataset_path: str):
        """Loads the serialized structures data from specified path, computes new edges for the structures based on the maximum number of neighbours and radius. Additionally,
        atom and structure features are updated.

        Parameters
        ----------
        dataset_path: str
            Directory path where files containing serialized structures are stored.
        Returns
        ----------
        [Data]
            List of Data objects representing atom structures.
        """
        dataset = []
        with open(dataset_path, "rb") as f:
            _ = pickle.load(f)
            _ = pickle.load(f)
            dataset = pickle.load(f)

        # FIXME: this assumes all structures will have the same edges.
        edge_index = self.__compute_edges(data=dataset[0])

        for data in tqdm(dataset):
            # FIXME: reusing positions/edges only works because each structure is identical.
            # data = self.__make_periodic(data, len(data.x), config["radius"])
            data.pos = dataset[0].pos
            data.edge_index = edge_index
            # data.edge_attr = edge_distances
            self.__update_predicted_values(
                self.config["Variables_of_interest"]["type"],
                self.config["Variables_of_interest"]["output_index"],
                data,
            )
            self.__update_atom_features(
                self.config["Variables_of_interest"]["input_node_features"], data
            )

        if "subsample_percentage" in self.config["Variables_of_interest"].keys():
            return self.__stratified_sampling(
                dataset=dataset,
                subsample_percentage=self.config["Variables_of_interest"][
                    "subsample_percentage"
                ],
            )

        return dataset

    def __update_atom_features(self, atom_features: [AtomFeatures], data: Data):
        """Updates atom features of a structure. An atom is represented with x,y,z coordinates and associated features.

        Parameters
        ----------
        atom_features: [AtomFeatures]
            List of features to update. Each feature is instance of Enum AtomFeatures.
        data: Data
            A Data object representing a structure that has atoms.
        """
        feature_indices = [i for i in atom_features]
        data.x = data.x[:, feature_indices]

    def __update_predicted_values(self, type: list, index: list, data: Data):
        """Updates values of the structure we want to predict. Predicted value is represented by integer value.
        Parameters
        ----------
        type: "graph" level or "node" level
        index: index/location in data.y for graph level and in data.x for node level
        data: Data
            A Data object representing a structure that has atoms.
        """
        output_feature = []
        for item in range(len(type)):
            if type[item] == "graph":
                feat_ = torch.reshape(data.y[index[item]], (1, 1))
            elif type[item] == "node":
                feat_ = torch.reshape(data.x[:, index[item]], (-1, 1))
            else:
                raise ValueError("Unknown output type", type[item])
            output_feature.append(feat_)
        data.y = torch.cat(output_feature, 0)

    def __compute_edges(self, data: Data):
        """Computes edges of a structure depending on the maximum number of neighbour atoms that each atom can have
        and radius as a maximum distance of a neighbour.

        Parameters
        ----------
        data: Data
            A Data object representing a structure that has atoms.

        Returns
        ----------
        torch.tensor
            Tensor filled with pairs (atom1_index, atom2_index) that represent edges or connections between atoms within the structure.
        """
        # We do not build the adjacency matrix because the indexing for periodic atoms needs to reflect the local (not unique ghost) index.
        print("Compute edges of the structure.")
        self.num_local = len(data.x)

        print("Computing edge distances and adding candidate neighbours.")
        # Guess needed neighbor allocation. # FIXME: this does not scale well.
        edge_index = np.zeros((2, self.num_local ** 2 * 27), dtype=np.int64)
        edge_count = 0
        for i in tqdm(range(self.num_local)):
            for j in range(self.num_local):
                edge_count = self.__add_neighbor(
                    data, edge_index, edge_count, data.pos[i], data.pos[j], i, j
                )

        if self.periodic:
            print("Adding periodic edges.")
            edge_count = self.__make_periodic(data, edge_index, edge_count)

        # Remove uneeded allocated space.
        edge_index = edge_index[:, : edge_count - 1]
        # Convert to torch object.
        edge_index = torch.tensor(edge_index)

        # Normalize the lengths using min-max normalization
        # edge_lengths = (edge_lengths - min(edge_lengths)) / (
        #    max(edge_lengths) - min(edge_lengths)
        # )
        # FIXME: return edge lengths when used.
        return edge_index  # , edge_lengths

    def __make_periodic(self, data, edge_index, edge_count):
        # Get cell lengths from atomic data.
        cell = self.__get_cell(data.pos)

        # Loop over all surrounding cells and add ghost atoms.
        ghost_pos = []
        ghost_x = []
        periodic = [-1, 0, 1]
        for p1 in periodic:
            for p2 in periodic:
                for p3 in periodic:
                    p = [p1, p2, p3]
                    if p != [0, 0, 0]:
                        for j in range(self.num_local):
                            pj = [0, 0, 0]
                            # Calculate periodic shifted position.
                            for d in range(3):
                                pj[d] = float(data.pos[j][d]) + cell[d] * p[d]
                            # Add this ghost neighbor to any nearby local (real) atom.
                            for i in range(self.num_local):
                                edge_count = self.__add_neighbor(
                                    data, edge_index, edge_count, data.pos[i], pj, i, j
                                )
        return edge_count

    def __add_neighbor(self, data, edge_index, edge_count, pi, pj, i, j):
        distance = distance_3D(pi, pj)
        if distance <= self.radius:
            edge_index[0, edge_count] = i
            edge_index[1, edge_count] = j
            edge_count += 1
        return edge_count

    # FIXME: this is a hack that only works for cubic crystals - the dataset should hold this unit cell information directly.
    def __get_cell(self, local_atoms):
        cell = [0, 0, 0]
        max_cell = list(torch.amax(local_atoms, 0))
        for d in range(3):
            # Assume this is a perfect crystal and find atoms with a single bond length between them in each dimension.
            bond_length = 0.0
            bond_index = 0
            while bond_length < 1e-9:
                bond_length = abs(
                    local_atoms[bond_index][d] - local_atoms[bond_index + 1][d]
                )
                bond_index += 1
            cell[d] = max_cell[d].item() + bond_length.item()
        return cell

    def __stratified_sampling(self, dataset: [Data], subsample_percentage: float):
        """Given the dataset and the percentage of data you want to extract from it, method will
        apply stratified sampling where X is the dataset and Y is are the category values for each datapoint.
        In the case of the structures dataset where each structure contains 2 types of atoms, the category will
        be constructed in a way: number of atoms of type 1 + number of protons of type 2 * 100.

        Parameters
        ----------
        dataset: [Data]
            A list of Data objects representing a structure that has atoms.
        subsample_percentage: float
            Percentage of the dataset.

        Returns
        ----------
        [Data]
            Subsample of the original dataset constructed using stratified sampling.
        """
        unique_values = torch.unique(dataset[0].x[:, 0]).tolist()
        dataset_categories = []
        print("Computing the categories for the whole dataset.")
        for data in tqdm(dataset):
            frequencies = torch.bincount(data.x[:, 0].int())
            frequencies = sorted(frequencies[frequencies > 0].tolist())
            category = 0
            for index, frequency in enumerate(frequencies):
                category += frequency * (100 ** index)
            dataset_categories.append(category)

        subsample_indices = []
        subsample = []

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=subsample_percentage, random_state=0
        )

        for subsample_index, rest_of_data_index in sss.split(
            dataset, dataset_categories
        ):
            subsample_indices = subsample_index.tolist()

        for index in subsample_indices:
            subsample.append(dataset[index])

        return subsample
