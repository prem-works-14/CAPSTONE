# ======================
# Imports
# ======================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# Settings
# ======================
SEQ_LEN = 20
NUM_POINTS = 1000
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
P_LOSS_WEIGHT = 0.1
KAPPA = 0.01  # diffusion coefficient

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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(point_idx, dtype=torch.long)

# Randomly pick points
points_idx = np.random.choice(data_seq.shape[0], size=NUM_POINTS, replace=False)
dataset = ClimateDataset(data_seq, seq_len=SEQ_LEN, points_idx=points_idx)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# Precompute neighbor table
# ======================
neighbor_table = {}
for idx in range(lat_size * lon_size):
    lat_idx = idx // lon_size
    lon_idx = idx % lon_size
    neighbors = []
    for dlat, dlon in [(-1,0), (1,0), (0,-1), (0,1)]:
        nlat, nlon = lat_idx + dlat, lon_idx + dlon
        if 0 <= nlat < lat_size and 0 <= nlon < lon_size:
            neighbors.append(nlat * lon_size + nlon)
        else:
            neighbors.append(None)
    neighbor_table[idx] = neighbors

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
# Physics-Informed Loss
# ======================
def physics_loss(model, X_batch, points_idx_batch, full_data):
    X_batch.requires_grad_(True)
    outputs = model(X_batch)
    t_pred = outputs[:, 1]

    grads = torch.autograd.grad(outputs=t_pred, inputs=X_batch,
                                grad_outputs=torch.ones_like(t_pred),
                                create_graph=True, retain_graph=True)[0]
    dt_dt = grads[:, -1, 1]

    laplacians = []
    points_idx = points_idx_batch.cpu().numpy()

    for pidx in points_idx:
        neighbors = neighbor_table[pidx]
        center = full_data[pidx]
        lap = 0
        count = 0
        for nb in neighbors:
            if nb is not None:
                lap += full_data[nb][-1,1]
                count += 1
        if count > 0:
            laplacian = (lap - count * center[-1,1]) / count
            laplacians.append(laplacian)
        else:
            laplacians.append(0.0)

    laplacians = torch.tensor(laplacians, device=device)

    residual = dt_dt - KAPPA * laplacians
    physics_loss = torch.mean(residual**2)
    return physics_loss

# ======================
# Training Loop
# ======================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    physics_running_loss = 0

    for X_batch, y_batch, points_idx_batch in train_loader:
        X_batch, y_batch, points_idx_batch = X_batch.to(device), y_batch.to(device), points_idx_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        mse_loss = criterion(outputs, y_batch)

        p_loss = physics_loss(model, X_batch, points_idx_batch, data_seq)
        total_loss = mse_loss + P_LOSS_WEIGHT * p_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        physics_running_loss += p_loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Total Loss: {running_loss/len(train_loader):.4f} | Physics Loss: {physics_running_loss/len(train_loader):.6f}")

# ======================
# Evaluation
# ======================
model.eval()
predictions, actuals, points = [], [], []
with torch.no_grad():
    for X_batch, y_batch, points_idx_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.append(outputs.cpu().numpy())
        actuals.append(y_batch.numpy())
        points.append(points_idx_batch.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)
points = np.hstack(points)

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
# Heatmap visualization
# ======================
# Only for temperature (t)
true_map = np.zeros(lat_size * lon_size)
pred_map = np.zeros(lat_size * lon_size)

for i, idx in enumerate(points):
    true_map[points_idx[idx]] = actuals[i,1]
    pred_map[points_idx[idx]] = predictions[i,1]

true_map = true_map.reshape(lat_size, lon_size)
pred_map = pred_map.reshape(lat_size, lon_size)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

c1 = axs[0].imshow(true_map, cmap='coolwarm')
axs[0].set_title("True Temperature")
plt.colorbar(c1, ax=axs[0])

c2 = axs[1].imshow(pred_map, cmap='coolwarm')
axs[1].set_title("Predicted Temperature")
plt.colorbar(c2, ax=axs[1])

plt.tight_layout()
plt.show()
