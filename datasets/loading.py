"""Dataset loading."""
import json
import os

import numpy as np
import pickle

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
    "custom"
]


def load_custom_data():
    x = pickle.load(open(os.path.join(os.environ["HHC_HOME"], "feature_matrix.pkl"), "rb"))
    x = np.array(x, dtype=float)
    # y = np.array([i % 5 for i in range(x.shape[0])], dtype=int)
    y = np.zeros((x.shape[0], 1))
    mean = x.mean(0)
    mean = mean + 0.00000001
    std = x.std(0)
    print(mean, std)
    x = (x - mean) / std
    return x, y


def load_data(dataset, normalize=False):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    normalize = False
    if dataset in UCI_DATASETS:
        if dataset != "custom":
            x, y = load_uci_data(dataset)
        else:
            x, y = load_custom_data()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = np.triu(similarities) + np.triu(similarities).T
    similarities[np.diag_indices_from(similarities)] = 1.0
    similarities[similarities > 1.0] = 1.0
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
        "custom": ()
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    with open("a.json", "w+") as f:
        json.dump({
            "x": x,
            "y": y
        }, f)
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    print("X shape", x.shape)
    print("Y shape", y.shape)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    return x, y
