import os, numpy as np, json, pickle, pathlib
import pytest

from torch import tensor
from torch_geometric.data import Data

from data_utils import SerializedDataLoader


@pytest.mark.mpi_skip()
def pytest_periodic():
    config_file = "./examples/test_periodic.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    # Create two nodes with arbitrary values.
    data = Data()
    data.pos = tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    data.x = tensor([[3, 5, 7], [9, 11, 13]])
    data.y = tensor([[99]])
    minmax = np.zeros((2, 2, 3))

    serial_data_dir = "./serialized_dataset/"
    serial_data_path = os.path.join(serial_data_dir, "unit_test_periodic.pkl")
    if not os.path.exists(serial_data_dir):
        os.mkdir(serial_data_dir)

    with open(serial_data_path, "wb") as f:
        pickle.dump(minmax, f)
        pickle.dump(minmax, f)
        pickle.dump([data], f)

    loader = SerializedDataLoader(config["NeuralNetwork"])
    periodic_dataset = loader.load_serialized_data(serial_data_path)
    periodic_data = periodic_dataset[0]

    # Check that there's still two nodes.
    assert periodic_data.edge_index.size(0) == 2
    # Check that there's one "real" bond and 26 ghost bonds (for both nodes).
    assert periodic_data.edge_index.size(1) == 1 + 26 * 2

    # Check the nodes were not modified.
    for i in range(2):
        for d in range(3):
            assert periodic_data.pos[i][d] == data.pos[i][d]
        assert periodic_data.x[i][0] == data.x[i][0]
    assert periodic_data.y == data.y


if __name__ == "__main__":
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    pytest_periodic()
