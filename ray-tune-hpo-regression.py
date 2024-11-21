import torch
import math
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.model_selection import train_test_split
import numpy as np

from config import heart_rate_features_1, heart_rate_features_2, \
                    sleep_features_1, sleep_features_2, \
                    intensity_features_1, intensity_features_2, \
                    steps_features_1, steps_features_2,  temporal_features \
                    
features_1 =  heart_rate_features_1 + sleep_features_1 + intensity_features_1 + steps_features_1 + temporal_features
features =  features_1 + heart_rate_features_2 + sleep_features_2 + intensity_features_2 + steps_features_2


# -------------------------------------------------------------------------------------
# Positional Encoding: Adds positional information to input embeddings
# -------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Initialize positional encoding module.
        Args:
            d_model: Model's feature dimension.
            max_seq_length: Maximum sequence length supported.
            dropout: Dropout probability applied after adding positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer('pe', pe)  # Save as a non-learnable parameter

    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model).
        Returns:
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]  # Match sequence length
        return self.dropout(x)

# -------------------------------------------------------------------------------------
# Depthwise Separable Convolution: Efficient feedforward layer for Transformer
# -------------------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        """
        Implements depthwise separable convolution for feedforward layers.
        Args:
            d_model: Feature size of the model.
            dim_feedforward: Expanded dimension in feedforward layers.
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding=1)
        self.pointwise = nn.Conv1d(d_model, dim_feedforward, kernel_size=1)

    def forward(self, x):
        """
        Perform depthwise followed by pointwise convolution.
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model).
        Returns:
            Output tensor with expanded dimension (batch_size, seq_len, dim_feedforward).
        """
        x = x.transpose(1, 2)  # Convert to (batch_size, features, seq_len) for Conv1d
        x = self.depthwise(x)  # Independent convolution per channel
        x = self.pointwise(x)  # Combine outputs from depthwise
        return x.transpose(1, 2)  # Back to (batch_size, seq_len, features)

# -------------------------------------------------------------------------------------
# Linear Attention: A placeholder for implementing efficient attention mechanisms
# -------------------------------------------------------------------------------------
class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size):
        """
        Placeholder for linear attention mechanism.
        Args:
            d_model: Model's feature size.
            num_heads: Number of attention heads.
            kernel_size: Kernel size for attention computation.
        """
        super(LinearAttention, self).__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size

        # Simple linear layers for queries, keys, and values
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        """
        Compute attention using linear transformations.
        Args:
            queries, keys, values: Input tensors for attention.
        Returns:
            Attention outputs.
        """
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        attention_scores = torch.relu(Q @ K.transpose(-2, -1))  # Simplified scoring
        return attention_scores @ V  # Weighted values

# -------------------------------------------------------------------------------------
# Custom Encoder Layer: Combines attention and feedforward logic
# -------------------------------------------------------------------------------------
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, attention_type, attn_kernel_size, depthwise_separable_conv):
        """
        Custom Transformer encoder layer.
        Args:
            d_model: Model's feature size.
            num_heads: Number of attention heads.
            dim_feedforward: Expanded size in feedforward network.
            dropout: Dropout probability.
            attention_type: Type of attention to use (e.g., "scaled_dot_product").
            attn_kernel_size: Kernel size for attention (if applicable).
            depthwise_separable_conv: Use depthwise separable convolution in feedforward.
        """
        super(CustomEncoderLayer, self).__init__()

        # Select attention mechanism
        if attention_type == "scaled_dot_product":
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        elif attention_type == "linear_attention":
            self.attention = LinearAttention(d_model, num_heads, kernel_size=attn_kernel_size)
        elif attention_type == "multi_head_attention":
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # Feedforward network with optional depthwise separable convolutions
        if depthwise_separable_conv:
            self.feedforward = DepthwiseSeparableConv(d_model, dim_feedforward)
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Perform attention and feedforward operations with residual connections.
        Args:
            src: Input tensor with shape (batch_size, seq_len, d_model).
        Returns:
            Processed tensor.
        """
        # Attention block
        attn_output, _ = self.attention(src, src, src)  # Self-attention
        src = self.norm1(src + self.dropout(attn_output))  # Residual connection + normalization

        # Feedforward block
        ff_output = self.feedforward(src)
        src = self.norm2(src + self.dropout(ff_output))  # Residual connection + normalization

        return src

# -------------------------------------------------------------------------------------
# Transformer Model: Combines encoder layers and a regression head
# -------------------------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, d_model, dim_feedforward, dropout, max_seq_length, attention_type, attn_kernel_size, depthwise_separable_conv):
        """
        Transformer model with configurable attention and feedforward layers.
        Args:
            input_size: Number of input features per token.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            d_model: Feature size of the model.
            dim_feedforward: Size of feedforward layer.
            dropout: Dropout probability.
            max_seq_length: Maximum supported sequence length.
            attention_type: Attention mechanism to use.
            attn_kernel_size: Kernel size for linear attention.
            depthwise_separable_conv: Use depthwise separable convolutions.
        """
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Create a stack of encoder layers
        self.layers = nn.ModuleList([
            CustomEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_type=attention_type,
                attn_kernel_size=attn_kernel_size,
                depthwise_separable_conv=depthwise_separable_conv
            ) for _ in range(num_encoder_layers)
        ])

        # Regression head for sequence output
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        """
        Forward pass for the Transformer model.
        Args:
            x: Input tensor with shape (batch_size, seq_len, input_size).
        Returns:
            Final output tensor with shape (batch_size, 1).
        """
        x = self.input_projection(x)  # Linear projection to match d_model
        x = self.positional_encoding(x)  # Add positional encodings
        for layer in self.layers:
            x = layer(x)  # Pass through each encoder layer
        out = x[:, -1, :]  # Use the last token for regression
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        return self.fc5(out)  # Final regression output

# -------------------------------------------------------------------------------------
# Custom Loss Function: MAPE (Mean Absolute Percentage Error)
# -------------------------------------------------------------------------------------
def mape_loss(output, target):
    epsilon = 1e-8  # Small value to prevent division by zero
    return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100


# -------------------------------------------------------------------------------------
# Training Function: Includes advanced hyperparameter handling.
# -------------------------------------------------------------------------------------
optimizer_mapping = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}

def train_transformer_model(config, data_loader, val_loader):
    """
    Train the Transformer model with given configuration and datasets.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
        data_loader (DataLoader): PyTorch DataLoader for training data.
        val_loader (DataLoader): PyTorch DataLoader for validation data.
    """
    # Initialize the Transformer model with parameters from config
    model = TransformerModel(
        input_size=10,
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        d_model=config["d_model"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_seq_length=config["max_seq_length"],
        attention_type=config["attention_type"],
        key_dim_scaling=config["key_dim_scaling"],
        attn_kernel_size=config["attn_kernel_size"],
        depthwise_separable_conv=config["depthwise_separable_conv"],
        shared_weights=config["shared_weights"],
        stochastic_depth_rate=config["stochastic_depth_rate"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Always use the first GPU for the trial
    model = model.to(device)  # Move model to the selected GPU

    # Define the optimizer based on the config
    optimizer_class = optimizer_mapping[config["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config.get("momentum", 0.0)  # Only applicable for SGD/RMSprop
    )

    # Define learning rate scheduler with warmup and decay
    warmup_steps = config["warmup_steps"]
    total_steps = config["total_steps"]

    def lr_lambda(current_step):
        """
        Learning rate schedule with warmup and linear decay.
        """
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Map loss function from the configuration
    loss_function_mapping = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.SmoothL1Loss(),
        "mape": mape_loss  # Custom MAPE loss
    }
    criterion = loss_function_mapping[config["loss_function"]]

    # Training loop
    for epoch in range(20):  # Train for 20 epochs
        model.train()
        epoch_loss = 0.0
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            # Apply gradient clipping if enabled
            if config["gradient_clipping"] > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])

            # Update model parameters
            optimizer.step()

            # Track epoch loss
            epoch_loss += loss.item()

        # Step the scheduler after each epoch
        scheduler.step()

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(data_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_mape = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute validation loss
            val_loss += criterion(outputs.squeeze(), targets).item()

            # Compute validation MAPE
            epsilon = 1e-7
            mape = torch.mean(torch.abs((targets - outputs.squeeze()) / (targets + epsilon))) * 100
            val_mape += mape.item()

    # Report metrics to Ray Tune
    tune.report(validation_loss=val_loss / len(val_loader), validation_mape=val_mape / len(val_loader))


# -------------------------------------------------------------------------------------
# Hyperparameter Search Space for Ray Tune
# -------------------------------------------------------------------------------------
search_space = {
    "num_heads": tune.choice([2, 4, 8, 16, 32]),
    "num_encoder_layers": tune.choice([2, 4, 6, 8, 12, 16]),
    "d_model": tune.choice([64, 128, 192, 256, 320, 512]),
    "dim_feedforward": tune.sample_from(lambda spec: spec.config["d_model"] * tune.choice([2, 3, 4])),
    "dropout": tune.loguniform(0.01, 0.5),
    "learning_rate": tune.loguniform(1e-5, 5e-2),
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "weight_decay": tune.loguniform(1e-6, 1e-1),
    "max_seq_length": tune.choice([50, 100, 200, 500, 1000, 2000]),
    "attention_type": tune.choice(["scaled_dot_product", "multi_head_attention", "linear_attention"]),
    "key_dim_scaling": tune.choice([1.0, 0.5, 0.25]),
    "attn_kernel_size": tune.choice([3, 5, 7]),
    "depthwise_separable_conv": tune.choice([True, False]),
    "shared_weights": tune.choice([True, False]),
    "stochastic_depth_rate": tune.uniform(0.0, 0.2),
    "warmup_steps": tune.choice([100, 500, 1000, 2000]),
    "total_steps": tune.choice([10000, 20000, 50000, 100000]),
    "loss_function": tune.choice(["mse", "mae", "huber", "mape"]),
    "gradient_clipping": tune.uniform(0.0, 1.0),
    "optimizer": tune.choice(["adam", "adamw", "sgd", "rmsprop"]),
}

# Function to split data into intervals
def split_into_intervals(data, interval_size, stride):
    intervals = []
    num_intervals = (data.shape[0] - interval_size) // stride + 1
    for i in range(num_intervals):
        start_ix = i * stride
        end_ix = start_ix + interval_size
        interval = data[start_ix:end_ix]
        intervals.append(interval)
    return np.array(intervals)

# Function to load data from .npy files
def load_dataframe_from_npy(file_path):
    data_dict = np.load(file_path, allow_pickle=True).item()
    columns = data_dict['columns']
    data = data_dict['data']
    return pd.DataFrame(data, columns=columns)

# -------------------------------------------------------------------------------------
# Data Preparation Functions
# -------------------------------------------------------------------------------------
def get_data_loaders(id_):
    """
    Prepare data loaders for training and validation.

    Args:
        id_ (str): Identifier for the dataset.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # File paths
    features_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_combined_data.npy'
    labels_data_path = f'/home/rxb2495/Glucose-Forecasting/final_data/{id_}_main_data.npy'

    # Load data
    intervals = load_dataframe_from_npy(features_data_path)
    output_data = load_dataframe_from_npy(labels_data_path)

    # Preprocess features and labels
    intervals = intervals[features]
    intervals = intervals.loc[:, ~intervals.columns.duplicated()]
    output_data = output_data["Historic Glucose mg/dL"].values.astype(np.float32)
    intervals = intervals.astype(np.float32).values
    intervals = split_into_intervals(intervals, 96, 96)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(intervals, output_data, test_size=0.3, random_state=42)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader


# -------------------------------------------------------------------------------------
# Main Script for Hyperparameter Tuning
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders("MMCS0002")

    # Run hyperparameter tuning with Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_transformer_model, data_loader=train_loader, val_loader=val_loader),
        config=search_space,
        num_samples=50,  # Number of trials
        scheduler=ASHAScheduler(metric="validation_mape", mode="min"),
        search_alg=BayesOptSearch(metric="validation_mape", mode="min", random_search_steps=10),
        resources_per_trial={"cpu": 2, "gpu": 1},  # Assign 1 GPU per trial
        local_dir="/home/rxb2495/ray_results",  # Directory to save results
        verbose=1
    )

    print("Best hyperparameters found: ", analysis.best_config)
