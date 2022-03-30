from cmath import log
import os
import torch
import importlib
import argparse
import numpy as np
import pytorch_lightning as pl

from ocd_detection import lit_models

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

avail_gpus = min(1, torch.cuda.device_count())

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.data.OCDDataModule'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def _setup_parser():
    """Setup Python's argument parser"""
    parser = argparse.ArgumentParser(add_help=False)
    # Add trainer specific arguments such as --max_epochs, --gpus, etc
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic Arguments
    parser.add_argument("--data_class", type=str, default="OCDDataModule")
    parser.add_argument("--model_class", type=str, default="TLModel")

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"ocd_detection.data.{temp_args.data_class}")
    model_class = _import_class(f"ocd_detection.models.{temp_args.model_class}")

    # Get data and model specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    # model_group = parser.add_argument("Model Args")
    # model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    """
    Run an experiment

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=10 --num_workers=8 --gpus=1
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    # Init our data pipeline
    data_class = _import_class(f"ocd_detection.data.{args.data_class}")
    data = data_class(args)
    # Init our model
    model_class = _import_class(f"ocd_detection.models.{args.model_class}")
    model = model_class(data_config=data.config(), args=args)
    
    lit_model_class = lit_models.BaseLitModel
    lit_model = lit_model_class(args=args, model=model)

    # Init logger
    logger = pl.loggers.TensorBoardLogger("training/logs")

    progress_bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=20)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [progress_bar_callback, early_stopping_callback, model_checkpoint_callback]

    # Init a trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")
    # Train the model
    trainer.fit(lit_model, datamodule=data)
    # Evaluate the model on the test set
    trainer.test(lit_model, datamodule=data)

if __name__ == '__main__':
    main()