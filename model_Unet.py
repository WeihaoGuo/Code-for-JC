import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler # 标准化

import matplotlib.pyplot as plt #绘图
from matplotlib.animation import FuncAnimation
from matplotlib import cm # colormap
import matplotlib.gridspec as gridspec
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau # 用于动态调整学习率
from torch.optim.lr_scheduler import StepLR # 用于动态调整学习率
from tqdm import tqdm  # 导入 tqdm
import h5py
from scipy import io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) 
# ------------------------------------------------------- 超参数设置 -------------------------------------------------------
num_epochs = 200 # num_epochs
batch_size = 6 # batch_size
learning_rate = 1e-3 # learning_rate

#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        
#        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
#        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#        
#        self.pool = nn.AdaptiveAvgPool2d((1, 1))
#        
#        self.fc = nn.Linear(128, 1)
#        self.relu = nn.ReLU()

#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.relu(x)
        
#        x = self.conv2(x)
#        x = self.relu(x)
        
#        x = self.conv3(x)
#        x = self.relu(x)
        
#        x = self.conv4(x)
#        x = self.relu(x)
        
#        x = self.pool(x)
#        x = torch.flatten(x, 1)
        
#        x = self.fc(x)
        
#        return x
    
# ------------------------------------------------------- model -------------------------------------------------------

# 定义双重卷积
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
# Downscaling    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
# Upscaling

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# outConv
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        return output


# ------------------------------------------------------- import data -------------------------------------------------------

# 使用 h5py 读取 .mat 文件
# 为了使代码简洁，可以将读取部分封装成一个函数
def load_mat_data(file_path, var_name):
    with h5py.File(file_path,'r') as file:
        return file[var_name][:]

data_path = 'input_data.mat'

ws = load_mat_data(data_path, 'ws')
u = load_mat_data(data_path, 'u')
v = load_mat_data(data_path, 'v')
qa = load_mat_data(data_path, 'qa')
sst = load_mat_data(data_path, 'sst')
cld = load_mat_data(data_path, 'cld')
slp = load_mat_data(data_path, 'slp')
ta = load_mat_data(data_path, 'ta')
lon = load_mat_data(data_path, 'x')
lat = load_mat_data(data_path, 'y')
year = load_mat_data(data_path, 'year')
month = load_mat_data(data_path, 'month')
ws_rss = load_mat_data(data_path, 'ws_rss')
#ws=ws*(-1)


nan_indices1 = np.where(np.isnan(ws[:, :, :]))
nan_indices2 = np.where(np.isnan(u[:, :, :]))
nan_indices3 = np.where(np.isnan(v[:, :, :]))
nan_indices4 = np.where(np.isnan(qa[:, :, :]))
nan_indices5 = np.where(np.isnan(sst[:, :, :]))
nan_indices6 = np.where(np.isnan(cld[:, :, :]))
nan_indices7 = np.where(np.isnan(slp[:, :, :]))
nan_indices8 = np.where(np.isnan(ta[:, :, :]))

# print(nan_indices1)
# print(nan_indices2)
nan_indices1_set = set(zip(nan_indices1[0], nan_indices1[1], nan_indices1[2]))
nan_indices2_set = set(zip(nan_indices2[0], nan_indices2[1], nan_indices2[2]))
nan_indices3_set = set(zip(nan_indices3[0], nan_indices3[1], nan_indices3[2]))
nan_indices4_set = set(zip(nan_indices4[0], nan_indices4[1], nan_indices4[2]))
nan_indices5_set = set(zip(nan_indices5[0], nan_indices5[1], nan_indices5[2]))
nan_indices6_set = set(zip(nan_indices6[0], nan_indices6[1], nan_indices6[2]))
nan_indices7_set = set(zip(nan_indices7[0], nan_indices7[1], nan_indices7[2]))
nan_indices8_set = set(zip(nan_indices8[0], nan_indices8[1], nan_indices8[2]))

# 合并六个集合
combined_nan_indices_set = nan_indices1_set.union(nan_indices2_set).union(nan_indices3_set).union(nan_indices4_set).union(nan_indices5_set).union(nan_indices6_set).union(nan_indices7_set).union(nan_indices8_set)

# 转换回数组格式
nan_indices  = np.array(list(combined_nan_indices_set)).T
print(nan_indices)


ws[nan_indices[0], nan_indices[1], nan_indices[2]] = 0
u[nan_indices[0], nan_indices[1], nan_indices[2]]= 0
v[nan_indices[0], nan_indices[1], nan_indices[2]]= 0
qa[nan_indices[0], nan_indices[1], nan_indices[2]] = 0
sst[nan_indices[0], nan_indices[1], nan_indices[2]] = 0
cld[nan_indices[0], nan_indices[1], nan_indices[2]]= 0
slp[nan_indices[0], nan_indices[1], nan_indices[2]]= 0
ta[nan_indices[0], nan_indices[1], nan_indices[2]]= 0


#ws_rss[ii, nan_indices[0], nan_indices[1]]= 0

# 将 (840, 66, 126) 扩展为 (1, 840, 66, 126)
ws = np.expand_dims(ws, axis=0)
u = np.expand_dims(u, axis=0)
v = np.expand_dims(v, axis=0)
qa = np.expand_dims(qa, axis=0)
sst = np.expand_dims(sst, axis=0)
cld = np.expand_dims(cld, axis=0)
slp = np.expand_dims(slp, axis=0)
ta = np.expand_dims(ta, axis=0)
lon = np.expand_dims(lon, axis=0)
lat = np.expand_dims(lat, axis=0)
year = np.expand_dims(year, axis=0)
month = np.expand_dims(month, axis=0)
ws_rss = np.expand_dims(ws_rss, axis=0)

# 使用 np.concatenate 在第一个维度上串联数组
data = np.concatenate((ws, u, v, qa, sst, ta, cld, slp), axis=0)
#data = np.concatenate((ws, cld, slp, qa, sst, ta), axis=0)

data = np.transpose(data, (1, 0, 2, 3))
labels = np.transpose(ws_rss, (1, 0, 2, 3))


# 计算划分索引
#train_split_index = int(0.8 * len(data))  # 80% 训练集
#val_split_index = int(0.9 * len(data))    # 10% 验证集，剩下10%为测试集
train_split_index = 828  # 1990-2018 训练集
val_split_index = 840    # 2019 验证集，2020-2022为测试集
# 按顺序划分训练集、验证集和测试集
train_data = data[516:train_split_index]
train_labels = labels[516:train_split_index]

val_data = data[train_split_index:val_split_index]
val_labels = labels[train_split_index:val_split_index]

test_data = data[val_split_index:]
test_labels = labels[val_split_index:]

# 打印形状以验证划分
print("训练集数据形状:", train_data.shape)
print("训练集标签形状:", train_labels.shape)
print("验证集数据形状:", val_data.shape)
print("验证集标签形状:", val_labels.shape)
print("测试集数据形状:", test_data.shape)
print("测试集标签形状:", test_labels.shape)

# ------------------------------------------------------- standard -------------------------------------------------------

# 创建 StandardScaler 实例
scaler_data = StandardScaler()

# Step 1: Reshape 数据以符合 sklearn 要求 (每个样本在一行，每个特征在一列)
# 假设 train_data 的形状为 (batch_size, channels, height, width)
batch_size, channels, height, width = train_data.shape

# 将数据 reshape 为 (batch_size * height * width, channels) 的形状
train_data_reshaped = train_data.transpose(0, 2, 3, 1).reshape(-1, channels)
val_data_reshaped = val_data.transpose(0, 2, 3, 1).reshape(-1, channels)
test_data_reshaped = test_data.transpose(0, 2, 3, 1).reshape(-1, channels)

train_labels_reshaped = train_labels.transpose(0, 2, 3, 1).reshape(-1, train_labels.shape[1])
val_labels_reshaped = val_labels.transpose(0, 2, 3, 1).reshape(-1, val_labels.shape[1])
test_labels_reshaped = test_labels.transpose(0, 2, 3, 1).reshape(-1, test_labels.shape[1])

# Step 2: 对数据进行标准化
train_data_normalized = scaler_data.fit_transform(train_data_reshaped)
val_data_normalized = scaler_data.transform(val_data_reshaped)
test_data_normalized = scaler_data.transform(test_data_reshaped)

train_labels_normalized = scaler_data.fit_transform(train_labels_reshaped)
val_labels_normalized = scaler_data.transform(val_labels_reshaped)
test_labels_normalized = scaler_data.transform(test_labels_reshaped)

# Step 3: 将标准化后的数据 reshape 回原始形状
train_data_normalized = train_data_normalized.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
val_data_normalized = val_data_normalized.reshape(val_data.shape[0], height, width, channels).transpose(0, 3, 1, 2)
test_data_normalized = test_data_normalized.reshape(test_data.shape[0], height, width, channels).transpose(0, 3, 1, 2)

train_labels_normalized = train_labels_normalized.reshape(train_labels.shape[0], height, width, train_labels.shape[1]).transpose(0, 3, 1, 2)
val_labels_normalized = val_labels_normalized.reshape(val_labels.shape[0], height, width, val_labels.shape[1]).transpose(0, 3, 1, 2)
test_labels_normalized = test_labels_normalized.reshape(test_labels.shape[0], height, width, test_labels.shape[1]).transpose(0, 3, 1, 2)

# Step 4: 将标准化后的数据转换为 PyTorch 张量
train_data_normalized = torch.tensor(train_data_normalized, dtype=torch.float32)
train_labels_normalized = torch.tensor(train_labels_normalized, dtype=torch.float32)

val_data_normalized = torch.tensor(val_data_normalized, dtype=torch.float32)
val_labels_normalized = torch.tensor(val_labels_normalized, dtype=torch.float32)

test_data_normalized = torch.tensor(test_data_normalized, dtype=torch.float32)
test_labels_normalized = torch.tensor(test_labels_normalized, dtype=torch.float32)

# Step 5: 创建 PyTorch 数据集和数据加载器
train_dataset = TensorDataset(train_data_normalized, train_labels_normalized)
val_dataset = TensorDataset(val_data_normalized, val_labels_normalized)
test_dataset = TensorDataset(test_data_normalized, test_labels_normalized)

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# 检查标准化后的数据形状和统计信息
print(f'训练集数据形状: {train_data_normalized.shape}')
print(f'验证集数据形状: {val_data_normalized.shape}')
print(f'测试集数据形状: {test_data_normalized.shape}')
print(f'训练集标签形状: {train_labels_normalized.shape}')
print(f'验证集标签形状: {val_labels_normalized.shape}')
print(f'测试集标签形状: {test_labels_normalized.shape}')

# 打印标准化后的均值和标准差
print(f'训练集标准化后均值: {train_data_normalized.mean(dim=(0, 2, 3))}')
print(f'训练集标准化后标准差: {train_data_normalized.std(dim=(0, 2, 3))}')
print(f'标签标准化后均值: {train_labels_normalized.mean(dim=(0, 2, 3))}')
print(f'标签标准化后标准差: {train_labels_normalized.std(dim=(0, 2, 3))}')

# ------------------------------------------------------- train -------------------------------------------------------
#model = CNN()
model = UNet(n_channels = 8, n_classes = 1).to(device)

criterion = nn.MSELoss() # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3) # 使用Adam优化器

# 定义 ReduceLROnPlateau 调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) # 验证集损失在2轮内没有改善则降低学习率至原来的0.5倍
# scheduler = StepLR(optimizer, step_size=4, gamma=0.5)  
# 初始化存储每轮指标的列表
train_losses = []
train_mses = []
val_losses = []
val_mses = []

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    # 在训练循环中加入 tqdm 进度条
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到设备
        
        optimizer.zero_grad()  # 清零梯度
        
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数
        
        running_loss += loss.item()  # 累积损失
        
        # 存储训练数据和预测结果
        y_true_train.append(labels.cpu().detach().numpy())
        y_pred_train.append(outputs.cpu().detach().numpy())
    
    # 计算训练集 MSE
    y_true_train = np.concatenate(y_true_train, axis=0)
    y_pred_train = np.concatenate(y_pred_train, axis=0)
    mse_train = root_mean_squared_error(y_true_train.flatten(), y_pred_train.flatten())

    # 保存训练损失和 MSE
    train_losses.append(running_loss / len(train_loader))
    train_mses.append(mse_train)

    # 打印训练损失和指标
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Loss: {running_loss/len(train_loader)}, Train RMSE: {mse_train}')

    # 验证模型
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    y_true_val = []
    y_pred_val = []

    # 在验证循环中加入 tqdm 进度条
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到设备
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 存储验证数据和预测结果
            y_true_val.append(labels.cpu().detach().numpy())
            y_pred_val.append(outputs.cpu().detach().numpy())
    
    # 计算验证集 MSE
    y_true_val = np.concatenate(y_true_val, axis=0)
    y_pred_val = np.concatenate(y_pred_val, axis=0)
    mse_val = root_mean_squared_error(y_true_val.flatten(), y_pred_val.flatten())

    # 保存验证损失和 MSE
    val_losses.append(val_loss / len(val_loader))
    val_mses.append(mse_val)

    # 打印验证损失和指标
    print(f'Validation Loss: {val_loss/len(val_loader)}, Validation RMSE: {mse_val}')

    # 打印当前学习率
    print(f'Learning Rate1: {optimizer.param_groups[0]["lr"]}')
    
    # 更新学习率
    # scheduler.step()

    # 更新调度器
    scheduler.step(val_loss)
    
    # 打印更新后的学习率
    print(f'Learning Rate2: {optimizer.param_groups[0]["lr"]}')


# ------------------------------------------------------- test -------------------------------------------------------

# 测试模型
model.eval()  # 设置模型为评估模式
test_loss = 0.0
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到设备
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # 存储测试数据和预测结果
        y_true_test.append(labels.cpu().detach().numpy())
        y_pred_test.append(outputs.cpu().detach().numpy())

# 将列表转换为数组（前提是形状一致）
y_true_test_array = np.concatenate(y_true_test, axis=0)
y_pred_test_array = np.concatenate(y_pred_test, axis=0)

# 反标准化
batch_size, channels, height, width = y_true_test_array.shape

# 将数据 reshape 为 (batch_size * height * width, channels) 的形状
y_true_test_reshaped = y_true_test_array.transpose(0, 2, 3, 1).reshape(-1, channels)
y_pred_test_reshaped = y_pred_test_array.transpose(0, 2, 3, 1).reshape(-1, channels)

# Step 2: 使用 scaler_labels 进行反标准化
y_true_test_reshaped = scaler_data.inverse_transform(y_true_test_reshaped)
y_pred_test_reshaped = scaler_data.inverse_transform(y_pred_test_reshaped)

# Step 3: 将数据 reshape 回原始形状
y_true_test_array = y_true_test_reshaped.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
y_pred_test_array = y_pred_test_reshaped.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

io.savemat('ws_test_Unet_2020_2022.mat',{'pred1':y_pred_test_array,'ws_rss':y_true_test_array}) 