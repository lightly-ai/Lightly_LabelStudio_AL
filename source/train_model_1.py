import numpy as np
import torch
from dataset import get_dataloader
from helper import dump_lightly_predictions, load_data, prepare_training_data
from model import Model, get_optimizer, predict, train_model

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    dataset_classes = ["cloudy", "rain", "shine", "sunrise"]
    prepare_training_data("./annotation-0.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 25

    train_X, train_y, _ = load_data("./train.json")
    val_X, val_y, _ = load_data("./val.json")
    dataloaders = {
        "train": get_dataloader(dataset_classes, train_X, train_y, batch_size=3),
        "val": get_dataloader(dataset_classes, val_X, val_y, batch_size=50),
    }

    model = Model().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    model = train_model(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        dataloaders=dataloaders,
        device=device,
        model_checkpoint=0,
        num_epochs=epochs,
    )

    train_X, train_y, filenames = load_data("./full_train.json")
    dataloader = get_dataloader(
        dataset_classes, train_X, train_y, batch_size=50, shuffle=False
    )
    predictions = predict(model, device, dataloader)
    dump_lightly_predictions(filenames, predictions)


main()
