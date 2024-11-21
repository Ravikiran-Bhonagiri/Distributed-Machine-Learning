import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
import numpy as np
from datetime import datetime
import logging

# -------------------------------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------------------------------
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(
    filename=f"/home/rxb2495/ray_tune_workflow_sample_run_{current_time}.log",  # Log file
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)
logger = logging.getLogger(__name__)  # Logger instance

# -------------------------------------------------------------------------------------
# Dummy Dataset Creation
# -------------------------------------------------------------------------------------
def create_dummy_data(num_samples=1000, seq_len=50, input_size=10):
    """
    Create a dummy dataset for regression tasks.
    Args:
        num_samples: Number of data samples.
        seq_len: Length of each sequence.
        input_size: Number of input features per time step.
    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    logger.info("Creating dummy dataset...")
    X = np.random.rand(num_samples, seq_len, input_size).astype(np.float32)
    y = np.random.rand(num_samples).astype(np.float32)

    # Split into training and validation
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors and DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    logger.info("Dummy dataset created successfully.")
    return train_loader, val_loader

# -------------------------------------------------------------------------------------
# Simple Transformer Model for Testing
# -------------------------------------------------------------------------------------
class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        """
        Simple Transformer model for regression.
        Args:
            input_size: Number of input features.
            d_model: Feature size of the model.
            num_heads: Number of attention heads.
            num_layers: Number of encoder layers.
            dropout: Dropout rate.
        """
        super(SimpleTransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)  # Project input to d_model size
        x = self.transformer(x)  # Pass through Transformer
        x = x[:, -1, :]  # Use the last time step
        return self.fc(x).squeeze()  # Output regression value

# -------------------------------------------------------------------------------------
# Training Function for Ray Tune
# -------------------------------------------------------------------------------------
def train_dummy_model(config, data_loader, val_loader):
    """
    Training function for Ray Tune.
    Args:
        config: Hyperparameter configuration.
        data_loader: Training DataLoader.
        val_loader: Validation DataLoader.
    """
    logger.info(f"Starting training with config: {config}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformerModel(
        input_size=10,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(5):  # Short training for testing
        model.train()
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(data_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    logger.info(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    # Report validation loss
    tune.report(validation_loss=val_loss / len(val_loader))

# -------------------------------------------------------------------------------------
# Hyperparameter Search Space
# -------------------------------------------------------------------------------------
search_space = {
    "d_model": tune.choice([32, 64, 128]),
    "num_heads": tune.choice([2, 4]),
    "num_layers": tune.choice([1, 2, 3]),
    "dropout": tune.uniform(0.1, 0.5),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-2)
}

# -------------------------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting the Ray Tune workflow...")

    # Create dummy data
    train_loader, val_loader = create_dummy_data()

    # Run Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_dummy_model, data_loader=train_loader, val_loader=val_loader),
        config=search_space,
        num_samples=10,  # Number of trials
        scheduler=ASHAScheduler(metric="validation_loss", mode="min"),
        resources_per_trial={"cpu": 2, "gpu": 1},  # Run one trial per GPU
        local_dir="/home/rxb2495/ray_results",  # Save results locally
        verbose=1
    )

    # Print and log the best hyperparameters
    best_config = analysis.best_config
    logger.info(f"Best hyperparameters found: \n {best_config}")
    print("Best hyperparameters found: \n", best_config)
