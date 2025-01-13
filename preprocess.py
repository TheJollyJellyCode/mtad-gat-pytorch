from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from args import get_parser


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def process_individual_and_continue(dataset_folder):
    """
    Verarbeitet einen INDIVIDUAL-Datensatz: Listet Spalten auf, ermöglicht Auswahl,
    normalisiert ausgewählte Spalten und führt danach den Standard-Workflow aus.
    """
    individual_dir = path.join(dataset_folder, "INDIVIDUAL")
    if not path.exists(individual_dir):
        raise ValueError(f"Der Ordner {individual_dir} existiert nicht!")

    # Erstellen eines neuen individuellen Ordners
    existing_folders = [f for f in listdir(dataset_folder) if f.startswith('INDIVIDUAL') and path.isdir(path.join(dataset_folder, f))]
    next_index = len(existing_folders)  # Startet bei 0, falls keine vorhanden sind
    next_individual_dir = path.join(dataset_folder, f"INDIVIDUAL{next_index}")
    makedirs(next_individual_dir, exist_ok=True)
    output_folder = path.join(next_individual_dir, "processed")
    makedirs(output_folder, exist_ok=True)

    print("Vorhandene Dateien im Ordner INDIVIDUAL:")
    files = [f for f in listdir(individual_dir) if f.endswith('.csv')]
    for idx, file in enumerate(files):
        print(f"{idx + 1}: {file}")

    file_choice = int(input("Wählen Sie eine Datei (Nummer): ")) - 1
    if file_choice < 0 or file_choice >= len(files):
        raise ValueError("Ungültige Auswahl!")

    chosen_file = files[file_choice]
    input_file_path = path.join(individual_dir, chosen_file)

    data = pd.read_csv(input_file_path)
    print("\nVerfügbare Spalten:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}: {column}")

    selected_columns = input("Geben Sie die Nummern der zu behaltenden Spalten ein (z.B. 1,2,5): ")
    selected_indices = [int(x.strip()) - 1 for x in selected_columns.split(",")]

    columns_to_keep = [data.columns[idx] for idx in selected_indices]
    if "timestamp" not in columns_to_keep:
        columns_to_keep.insert(0, "timestamp")  # Timestamp sicherstellen

    reduced_data = data[columns_to_keep]

    # Initialisieren des Scalers
    scaler = MinMaxScaler()

    columns_to_normalize = [col for col in columns_to_keep if col != "timestamp"]
    normalized_data = scaler.fit_transform(reduced_data[columns_to_normalize])

    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
    normalized_df["timestamp"] = reduced_data["timestamp"].values

    # Speichern der normalisierten Daten und der Parameter
    normalized_data_path = path.join(output_folder, "INDIVIDUAL_normalized_data.csv")
    params_path = path.join(output_folder, "INDIVIDUAL_normalization_params.csv")
    normalized_df.to_csv(normalized_data_path, index=False)

    params = pd.DataFrame({
        'feature': columns_to_normalize,
        'min': scaler.data_min_,
        'max': scaler.data_max_
    })
    params.to_csv(params_path, index=False)

    print(f"Normalisierte Daten gespeichert unter: {normalized_data_path}")
    print(f"Normalisierungsparameter gespeichert unter: {params_path}")

    # Weiter mit dem Standard-Workflow
    print("Führe den Standard-Workflow für INDIVIDUAL fort...")

    # Zeitstempel und Train/Test-Split
    timestamps = normalized_df["timestamp"]
    normalized_df = normalized_df.drop(columns=["timestamp"], errors="ignore")

    split_point = int(len(normalized_df) * 0.8)
    train_data = normalized_df.iloc[:split_point]
    test_data = normalized_df.iloc[split_point:]
    timestamps_train = timestamps.iloc[:split_point]
    timestamps_test = timestamps.iloc[split_point:]

    # Speichere als `.pkl`
    train_data.to_pickle(path.join(output_folder, "INDIVIDUAL_train.pkl"))
    test_data.to_pickle(path.join(output_folder, "INDIVIDUAL_test.pkl"))
    timestamps_train.to_pickle(path.join(output_folder, "INDIVIDUAL_timestamps_train.pkl"))
    timestamps_test.to_pickle(path.join(output_folder, "INDIVIDUAL_timestamps_test.pkl"))

    print("INDIVIDUAL-Datensatz erfolgreich verarbeitet und gespeichert.")


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """
    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

    elif dataset == "MYDATA":
        dataset_folder = "datasets/MYDATA"
        output_folder = "datasets/MYDATA/processed"
        makedirs(output_folder, exist_ok=True)
        data = pd.read_csv(path.join(dataset_folder, "daten_standard.csv"))

        timestamps = data["timestamp"]
        data_numeric = data.drop(columns=["timestamp"], errors="ignore")

        split_point = int(len(data_numeric) * 0.8)
        train_data = data_numeric.iloc[:split_point]
        test_data = data_numeric.iloc[split_point:]
        timestamps_train = timestamps.iloc[:split_point]
        timestamps_test = timestamps.iloc[split_point:]

        train_data.to_pickle(path.join(output_folder, "MYDATA_train.pkl"))
        test_data.to_pickle(path.join(output_folder, "MYDATA_test.pkl"))
        timestamps_train.to_pickle(path.join(output_folder, "MYDATA_timestamps_train.pkl"))
        timestamps_test.to_pickle(path.join(output_folder, "MYDATA_timestamps_test.pkl"))

    elif dataset == "INDIVIDUAL":
        dataset_folder = "datasets"
        process_individual_and_continue(dataset_folder)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
