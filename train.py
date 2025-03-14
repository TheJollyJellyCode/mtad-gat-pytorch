import json
from datetime import datetime
import os
import torch.nn as nn

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    args_summary = str(args.__dict__)
    print(args_summary)

    # Dataset-Spezifikationen
    if dataset == 'SMD':
        output_path = os.path.join('output', 'SMD', args.group)
        (x_train, _), (x_test, y_test) = get_data(f"machine-{args.group[0]}-{args.group[2:]}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = os.path.join('output', dataset)
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset == 'MYDATA':
        output_path = os.path.join('output', 'MYDATA')
        (x_train, timestamps_train), (x_test, timestamps_test, y_test) = get_data("MYDATA", normalize=normalize)
    elif dataset =='INDIVIDUAL1':
        output_path = f'output/INDIVIDUAL1'
        (x_train, timestamps_train), (x_test, timestamps_test, y_test) = get_data("INDIVIDUAL1", normalize=normalize)
    elif dataset =='INDIVIDUAL2':
        output_path = f'output/INDIVIDUAL2'
        (x_train, timestamps_train), (x_test, timestamps_test, y_test) = get_data("INDIVIDUAL2", normalize=normalize)
    elif dataset =='INDIVIDUAL3':
        output_path = f'output/INDIVIDUAL3'
        (x_train, timestamps_train), (x_test, timestamps_test, y_test) = get_data("INDIVIDUAL3", normalize=normalize)
    elif dataset =='INDIVIDUAL4':
        output_path = f'output/INDIVIDUAL4'
        (x_train, timestamps_train), (x_test, timestamps_test, y_test) = get_data("INDIVIDUAL4", normalize=normalize)
    else:
        raise ValueError(f'Dataset "{dataset}" not recognized. Please use one of: SMD, MSL, SMAP, MYDATA, INDIVIDUALx.')

    # Logging und Speicherpfade
    print(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")
    print(f"Number of features: {x_train.shape[1]}, Window size: {window_size}")
    log_dir = os.path.join(output_path, "logs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = os.path.join(output_path, id)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    # Zielgrößen und Modellinitialisierung
    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    print(
        f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)
    print("Starting training...")
    plot_losses(trainer.losses, save_path=save_path, plot=False)
    print("Training completed.")

    # Testen und Evaluierung
    test_loss = trainer.evaluate(test_loader)
    print("Starting evaluation on test data...")
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Vorschläge für Parameter
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "MYDATA": (0.95, 0.001),  # Beispielwerte für deinen Datensatz
        "INDIVIDUAL1": (0.95, 0.001),
        "INDIVIDUAL2": (0.95, 0.001),
        "INDIVIDUAL3": (0.95, 0.001),
        "INDIVIDUAL4": (0.95, 0.001),

    }
    # Dynamische Unterstützung für INDIVIDUALx
    if dataset.startswith("INDIVIDUAL"):
        level_q_dict[dataset] = (0.95, 0.001)

    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict.get(key, (0.95, 0.001))


    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "MYDATA": 0, "INDIVIDUAL1":0, "INDIVIDUAL2":0, "INDIVIDUAL3":0, "INDIVIDUAL4":0}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(os.path.join(save_path, "model.pt"))
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Speichern der Konfiguration
    args_path = os.path.join(save_path, "config.txt")
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
