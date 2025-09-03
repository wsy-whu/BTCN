import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pyro
from pyro.nn import PyroSample, PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroOptim
from pyro.distributions import Normal
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('Data//Phen_combined_xy_normalized.csv')

# 初始化一个空的列表用于存储整理后的数据
X = []
y = []

# 根据codeID进行分组
groups = df.groupby('codeID')

# 列表包含的特征名
feature_cols = ['EVI', 'NDVI', 'Qair_f_inst', 'SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'Tem_Nonzero',
                'evaporation_from_the_top_of_canopy_sum',
                'evaporation_from_vegetation_transpiration_sum', 'potential_evaporation_sum',
                'soil_temperature_level_1', 'soil_temperature_level_2', 'surface_net_solar_radiation_sum',
                'surface_pressure', 'temperature_2m_max', 'temperature_2m_min', 'total_evaporation_sum',
                'total_precipitation_sum', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2']

year = 17
X_train, X_test = [], []
y_train, y_test = [], []
y_test_codeID = []
for _, group in groups:
    feature_group = group[feature_cols].values.reshape(6, -1)  # 将特征转换为6*len(feature_cols)的矩阵
    target_value = group['kg_ha'].iloc[0]  # 假设每个codeID的kg_ha值相同，只取第一个

    # 检查 codeID 的前两位数字是否为 12
    if (group['codeID'].iloc[0] // 1000000) == year:
        X_test.append(feature_group)
        y_test.append(target_value)
        y_test_codeID = group['codeID'].iloc[0]
    else:
        X_train.append(feature_group)
        y_train.append(target_value)

print('X_train.shape', np.array(X_train).shape)
print('y_train.shape', np.array(y_train).shape)
print('X_test.shape', np.array(X_test).shape)
print('y_test.shape', np.array(y_test).shape)

# 标准化训练和测试数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.array(X_train).reshape(-1, 19))
X_test_scaled = scaler.transform(np.array(X_test).reshape(-1, 19))

X_train_tensor = torch.Tensor(X_train_scaled.reshape(-1, 6, 19)).permute(0, 2, 1)
X_test_tensor = torch.Tensor(X_test_scaled.reshape(-1, 6, 19)).permute(0, 2, 1)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

y_train_tensor = torch.Tensor(y_train_log).unsqueeze(1)
y_test_tensor = torch.Tensor(y_test_log).unsqueeze(1)


# X_train_tensor = torch.Tensor(np.array(X_train)).permute(0,2,1)
# y_train_tensor = torch.Tensor(np.array(y_train)).unsqueeze(1)
# X_test_tensor = torch.Tensor(np.array(X_test)).permute(0,2,1)
# y_test_tensor = torch.Tensor(np.array(y_test)).unsqueeze(1)

# 创建训练和测试数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 64
# 加载数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BayesianTCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size, dropout=dropout)
        self.mean = nn.Linear(num_channels[-1], 1)
        self.sigma = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1]  # 取序列的最后一个输出
        mean = self.mean(x).squeeze()
        sigma = torch.exp(self.sigma(x)).squeeze()  # 使用 exp 保证标准差为正
        return mean, sigma

    def loss_function(self, y_pred, y_true):
        mean, sigma = y_pred
        dist = Normal(mean, sigma)
        loss = -dist.log_prob(y_true).mean()  # 负对数似然
        return loss

# 同前面的 TemporalConvNet 和 TemporalBlock 类定义
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.5):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.conv1(x)))
    
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 模型、优化器初始化和训练过程
model = BayesianTCN(input_channels=19, num_channels=[64, 32, 64], kernel_size=3, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, optimizer, train_loader, num_epochs, early_stopping=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.loss_function(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {average_loss}')
        if early_stopping:
            early_stopping(average_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

early_stopping = EarlyStopping(patience=5, verbose=True)
train(model, optimizer, train_loader, num_epochs=200, early_stopping=early_stopping)

# 评估函数，包括计算均值和标准差
def evaluate(model, test_loader):
    model.eval()
    prediction_mean = []
    prediction_stdv = []
    with torch.no_grad():
        for x, _ in test_loader:
            mean, sigma = model(x)
            prediction_mean.extend(mean.tolist())
            prediction_stdv.extend(sigma.tolist())
    return prediction_mean, prediction_stdv

prediction_mean, prediction_stdv = evaluate(model, test_loader)
# predictions = np.array(predictions)
prediction_mean = np.array(np.exp(prediction_mean))
r2 = r2_score(y_test, prediction_mean)
rmse = sqrt(mean_squared_error(y_test, prediction_mean))

print(f'R^2: {r2:.3f}')
print(f'RMSE: {rmse:.3f}')
predicted_stds = []
predicted_means = []

for _ in range(100):
    prediction_mean, prediction_stdv = evaluate(model, test_loader)
    prediction_mean = np.array(np.exp(prediction_mean))
    predicted_means.append(prediction_mean)
    predicted_stds.append(prediction_stdv)

sigma_model =np.std(predicted_means,axis=0) # the epistemic uncertainty
sigma_data =np.mean(predicted_stds,axis=0) # the aleatoric uncertainty

print('averaged epistemic uncertainty',np.mean(sigma_model))
print('averaged aleatoric uncertainty',np.mean(sigma_data))