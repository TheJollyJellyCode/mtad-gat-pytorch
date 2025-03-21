import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    elif dataset == "MYDATA":
        # Passe die Anzahl der Features deines Datensatzes an
        return 14  # Beispiel: 15 Features
    elif dataset == "INDIVIDUAL1" : return 5
    elif dataset == "INDIVIDUAL2":  return 4
    elif dataset == "INDIVIDUAL3":
        return 4
    elif dataset == "INDIVIDUAL4": return 3
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    elif dataset == "MYDATA":
        return None  # Beispiel: Alle Features modellieren
    elif str(dataset).startswith("INDIVIDUAL"):
        return None



def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files.

    Parameters:
        dataset (str): Name of the dataset (e.g., SMD, MYDATA, INDIVIDUAL0, INDIVIDUAL1, etc.).
        max_train_size (int, optional): Maximum number of training samples. Defaults to None.
        max_test_size (int, optional): Maximum number of testing samples. Defaults to None.
        normalize (bool, optional): Whether to normalize the data. Defaults to False.
        spec_res (bool, optional): Specific resolution flag (not used). Defaults to False.
        train_start (int, optional): Start index for training data. Defaults to 0.
        test_start (int, optional): Start index for testing data. Defaults to 0.

    Returns:
        tuple: ((train_data, timestamps_train), (test_data, timestamps_test, test_label))
    """
    prefix = "datasets"

    # Set the correct prefix for each dataset
    if str(dataset).startswith("machine"):
        prefix = os.path.join(prefix, "ServerMachineDataset", "processed")
    elif dataset in ["MSL", "SMAP"]:
        prefix = os.path.join(prefix, "data", "processed")
    elif dataset == "MYDATA":
        prefix += "/MYDATA/processed"
    elif dataset == "INDIVIDUAL1" :
        prefix += "/INDIVIDUAL1/processed"
    elif dataset == "INDIVIDUAL2":
        prefix += "/INDIVIDUAL2/processed"
    elif dataset == "INDIVIDUAL3":
        prefix += "/INDIVIDUAL3/processed"
    elif dataset == "INDIVIDUAL4" :
        prefix += "/INDIVIDUAL4/processed"

    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size

    print(f"Loading data for dataset: {dataset}")
    print(f"Train range: {train_start} to {train_end}")
    print(f"Test range: {test_start} to {test_end}")

    x_dim = get_data_dim(dataset)  # Dynamically determine the number of features

    # Load training data
    train_data_path = os.path.join(prefix, f"{dataset}_train.pkl")
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f).values.reshape((-1, x_dim))[train_start:train_end, :]

    # Load testing data
    try:
        test_data_path = os.path.join(prefix, f"{dataset}_test.pkl")
        with open(test_data_path, "rb") as f:
            test_data = pickle.load(f).values.reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None

    # Load test labels
    try:
        test_label_path = os.path.join(prefix, f"{dataset}_test_label.pkl")
        with open(test_label_path, "rb") as f:
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    # Load timestamps for training and testing
    try:
        timestamps_train_path = os.path.join(prefix, f"{dataset}_timestamps_train.pkl")
        with open(timestamps_train_path, "rb") as f:
            timestamps_train = pickle.load(f)

        timestamps_test_path = os.path.join(prefix, f"{dataset}_timestamps_test.pkl")
        with open(timestamps_test_path, "rb") as f:
            timestamps_test = pickle.load(f)
    except (KeyError, FileNotFoundError):
        timestamps_train = None
        timestamps_test = None

    # Normalize data if specified
    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    # Print shapes of the data for debugging
    print("Train set shape: ", train_data.shape)
    print("Test set shape: ", test_data.shape if test_data is not None else "None")
    print("Test set label shape: ", None if test_label is None else test_label.shape)

    return (train_data, timestamps_train), (test_data, timestamps_test, test_label)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                      np.array([i - buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
        e_s = adjusted_scores[c_start: c_end + 1]

        e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end + 1] = e_s

    return adjusted_scores