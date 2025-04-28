import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ======================
# Settings
# ======================
SEQ_LEN = 20
NUM_POINTS = 1000
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# Load Data
# ======================
ds = xr.open_dataset("C:/Users/Dell/Downloads/climax_training_setup/data/sample/era5.nc")
z = ds["z"].values.squeeze()
t = ds["t"].values.squeeze()

# Normalize
z_mean, z_std = np.mean(z), np.std(z)
t_mean, t_std = np.mean(t), np.std(t)
z_norm = (z - z_mean) / z_std
t_norm = (t - t_mean) / t_std

# Combine variables
data = np.stack([z_norm, t_norm], axis=-1)
time_steps, lat_size, lon_size, var_count = data.shape
data_seq = data.reshape(time_steps, -1, 2).transpose(1, 0, 2)  # (num_points, time_steps, 2)

# ======================
# Dataset Class
# ======================
class ClimateDataset(Dataset):
    def __init__(self, data_seq, seq_len, points_idx):
        self.data_seq = data_seq
        self.seq_len = seq_len
        self.points_idx = points_idx
        self.time_steps = data_seq.shape[1]

    def __len__(self):
        return (self.time_steps - self.seq_len - 1) * len(self.points_idx)

    def __getitem__(self, idx):
        point_idx = idx % len(self.points_idx)
        time_idx = idx // len(self.points_idx)

        point = self.points_idx[point_idx]
        x = self.data_seq[point, time_idx:time_idx+self.seq_len, :]
        y = self.data_seq[point, time_idx+self.seq_len, :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Randomly pick some points
points_idx = np.random.choice(data_seq.shape[0], size=NUM_POINTS, replace=False)

dataset = ClimateDataset(data_seq, seq_len=SEQ_LEN, points_idx=points_idx)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# Model
# ======================
class BiLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = BiLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================
# Training Loop
# ======================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# ======================
# Evaluation
# ======================
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.append(outputs.cpu().numpy())
        actuals.append(y_batch.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# ======================
# Metrics
# ======================
for i, var in enumerate(['z', 't']):
    mse = mean_squared_error(actuals[:, i], predictions[:, i])
    mae = mean_absolute_error(actuals[:, i], predictions[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals[:, i], predictions[:, i])
    print(f"\n📊 Variable: {var}")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# ======================
# Plotting
# ======================
plt.plot(actuals[:100, 0], label='True z')
plt.plot(predictions[:100, 0], label='Pred z')
plt.legend()
plt.show()
