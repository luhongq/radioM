import torch
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import KFold,train_test_split
from AlexNet import AlexNetDynamic as AlexNet
from AlexNet import FullModel
# from radio import RadioWNet as AlexNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

#radioM数据集loader
class FULLmodelDataset(Dataset):
    def __init__(self, transform=None):

        self.image_path= r'F:\wireless\PathLossPredictionSatelliteImages-master\raw_data\mapbox_api'
        self.params = pd.read_csv(r'F:\wireless\PathLossPredictionSatelliteImages-master\raw_data\result.csv')
        self.transform = transform


    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        # 获取图像路径
        path_image = f"{self.image_path}/{idx}.png"

        # 打开图像
        image = Image.open(path_image)

        # 如果提供了 transform，应用 transform（如数据增强等）
        if self.transform:
            image = [self.transform(image)]

        # 获取该索引对应的系统参数
        selected_row = self.params.iloc[idx]

        # 读取并转换系统参数
        sys_params = [
            selected_row['L'],  # 假设是距离
            selected_row['FB'],  # 假设是频带
            selected_row['D'],  # 假设是其他特征
            selected_row['RSP']  # 假设是系统参数
        ]
        sys_params = list(map(float, sys_params))  # 转换为浮点数列表
        sys_params = torch.tensor(sys_params, dtype=torch.float32)  # 转换为PyTorch张量


        # 获取标签（例如 RSRP）
        label = torch.tensor(selected_row['RSRP'], dtype=torch.float32)

        # 返回图像、系统参数、标签和通道数（假设为3）
        return {
            "images": image,
            "sys_params": sys_params,
            "label": label,
            "channels": 3  # 这里假设图像是RGB的，所以通道数为3
        }
#实测数据集loader
class PropagationDataset(Dataset):
    def __init__(self, baseids, transform=None, rx_type=['img'], tx_type=['img']):
        """
        Args:
            baseids (list): 包含所有 baseid 的列表。
            transform (callable, optional): 对图像数据进行的预处理。
            rx_type (list): Rx 图像的类型列表。
            tx_type (list): Tx 图像的类型列表。
        """
        self.baseids = baseids
        self.image_dirs = {baseid: f"dataset/{baseid}/" for baseid in baseids}
        self.params_files = {baseid: pd.read_csv(f"dataset/{baseid}/{baseid}.csv") for baseid in baseids}
        self.transform = transform
        self.rx_type = rx_type
        self.tx_type = tx_type
        self.num = len(tx_type) + len(rx_type)

        # 计算每个 baseid 数据集的起始索引和长度
        self.baseid_ranges = {}
        current_start_idx = 0
        for baseid in baseids:
            length = len(self.params_files[baseid])
            self.baseid_ranges[baseid] = (current_start_idx, current_start_idx + length)
            current_start_idx += length


    def __len__(self):
        return sum(len(params) for params in self.params_files.values())

    def __getitem__(self, idx):

        # 根据全局索引找到对应的 baseid
        for baseid, (start_idx, end_idx) in self.baseid_ranges.items():

            if start_idx <= idx < end_idx:
                relative_idx = idx - start_idx  # 转换为相对索引
                params_data = self.params_files[baseid]
                image_dir = self.image_dirs[baseid]
                selected_row = params_data.iloc[relative_idx]

                break
        else:
            raise IndexError(f"Index {idx} out of range.")

        images = []
        channel_counts = 0

        # 加载 Tx 图像
        for type in self.tx_type:
            path_tx = f"{image_dir}/{type}/tx/0.tiff"
            try:
                image = Image.open(path_tx)
                channel_counts += len(image.getbands())
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except FileNotFoundError:
                print(f"Warning: File not found at {path_tx}")
                continue

        # 加载 Rx 图像
        for name_type in self.rx_type:
            path_rx = f"{image_dir}/{name_type}/rx/{relative_idx + 1}.tiff"
            try:
                image = Image.open(path_rx)
                channel_counts += len(image.getbands())
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except FileNotFoundError:
                print(f"Warning: File not found at {path_rx}")
                continue


        # 读取系统参数
        sys_params = [
            selected_row['L'], selected_row['FB'],selected_row['D'],selected_row['RSP']
        ]


       
        sys_params = list(map(float, sys_params))
        sys_params = torch.tensor(sys_params, dtype=torch.float32)



        return {
            "images": images,
            "sys_params": sys_params,
            "label": torch.tensor(selected_row['RSRP'], dtype=torch.float32),
            "channels": channel_counts
        }

import csv

# 定义保存训练和验证损失为 CSV 文件的函数
def save_loss_history(train_loss_list, val_loss_list, num_epochs, model):
    """
    保存训练和验证损失为 CSV 文件。

    Args:
        train_loss_list (list): 每个 epoch 的训练损失列表。
        val_loss_list (list): 每个 epoch 的验证损失列表。
        num_epochs (int): 总训练轮数。
        tx_list (list): tx 参数列表。
        rx_list (list): rx 参数列表。
    """

    file_name = f"{model}_{num_epochs}.csv"
    file_path = os.path.join("history", file_name)

    # 确保目录存在
    os.makedirs("history", exist_ok=True)

    # 写入 CSV 文件
    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_loss_list, val_loss_list), start=1):
            writer.writerow([epoch, train_loss, val_loss])

    print(f"Loss history saved to {file_path}")

def train_validate_test_split(dataset, test_size=0.2, random_state=42):
    """
    划分数据集为训练+验证集和测试集。

    Args:
        dataset: 数据集实例。
        test_size: 测试集比例。
        random_state: 随机种子。

    Returns:
        train_val_indices, test_indices: 训练+验证集索引和测试集索引。
    """
    total_size = len(dataset)
    indices = np.arange(total_size)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    return train_val_indices, test_indices

def cross_validate_with_test(dataset, model, criterion, optimizer, num_epochs=100, k_folds=5, batch_size=32, test_size=0.2, device="cuda",save_path='model/deep/airrx.pth',model_name='FusionNet'):
    """
    使用 KFold 对训练集交叉验证，并在测试集上评估模型。

    Args:
        dataset: 数据集实例。
        model: 待训练的模型。
        criterion: 损失函数。
        optimizer: 优化器。
        num_epochs: 每个折的训练 epoch 数。
        k_folds: 交叉验证折数。
        batch_size: 每个 batch 的大小。
        test_size: 测试集比例。
        device: 使用的设备（'cuda' 或 'cpu'）。

    Returns:
        None
    """
    # 划分数据集


    train_val_indices, test_indices = train_validate_test_split(dataset, test_size=test_size)
    train_val_dataset = Subset(dataset, train_val_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f'dataset;{len(dataset)},train:{len(train_val_dataset)},test;{len(test_dataset)}')
    # 初始化 KFold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    epochs=0
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_val_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # 训练集和验证集
        train_subset = Subset(train_val_dataset, train_indices)
        val_subset = Subset(train_val_dataset, val_indices)

        # 数据加载器
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


        # 将模型移动到设备
        model.to(device)

        for epoch in range(num_epochs):
            epochs+=1
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_rmse_sum=0.0
            for batch in tqdm(train_loader):

                images = torch.cat([img.to(device) for img in batch["images"]],dim=1)

                sys_params = batch["sys_params"].to(device)

                labels = batch["label"].to(device)




                outputs = model(images, sys_params)
                loss = criterion(outputs.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # 累加平方误差
                train_rmse_sum += ((outputs.squeeze() - labels) ** 2).sum().item()

            # 计算训练集的 RMSE
            train_rmse = (train_rmse_sum / len(train_loader.dataset)) ** 0.5

            avg_train_loss = train_loss / len(train_loader)
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f},RMSE: {train_rmse:.4f}")

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_rmse_sum = 0.0  # RMSE 累加器
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    images = torch.cat([img.to(device) for img in batch["images"]],dim=1)
                    sys_params = batch["sys_params"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(images, sys_params)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
                    # 计算 RMSE 的累加部分
                    val_rmse_sum += ((outputs.squeeze() - labels) ** 2).sum().item()



            avg_val_loss = val_loss / len(val_loader)
            # 计算验证集的 RMSE
            val_rmse = (val_rmse_sum / len(val_loader.dataset)) ** 0.5
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f},RMSE: {val_rmse:.4f}")
            # 如果当前验证集损失是最小的，则保存模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model, save_path)
                print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
            train_losses.append(train_rmse)
            val_losses.append(val_rmse)


        # 保存每个折的验证集平均损失
        fold_results.append(avg_val_loss)

    save_loss_history(train_losses, val_losses, epochs, model=model_name)
    # 打印交叉验证结果
    print("\nCross-Validation Results:")
    for fold_idx, loss in enumerate(fold_results):
        print(f"Fold {fold_idx + 1}: Validation Loss = {loss:.4f}")
    print(f"Average Validation Loss: {np.mean(fold_results):.4f}")

    # 加载验证阶段保存的最佳模型
    model=torch.load(save_path)
    model.to(device)
    print(f"\nLoaded the best model from {save_path} for evaluation.")

    # 在测试集上评估模型
    print("\nEvaluating on Test Set...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    test_loss = 0.0
    test_rmse_sum=0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = torch.cat([img.to(device) for img in batch["images"]],dim=1)
            sys_params = batch["sys_params"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, sys_params)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            test_rmse_sum += ((outputs.squeeze() - labels) ** 2).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_rmse= (test_rmse_sum / len(test_loader.dataset)) ** 0.5
    print(f"Test Set Loss: {avg_test_loss:.4f},Test Set Loss:(dBm):{test_rmse:.4f}")



if __name__ == "__main__":
    # 数据预处理 - 图像标准化
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化
        transforms.Lambda(lambda x: x.to(torch.float32))

    ])

    # baseid = [751630,524287]
    


    # # 实测数据集导入
    # rx_type = ['imgw']
    # tx_type = []

    # 创建Dataset和DataLoader

    # dataset = PropagationDataset(baseids=baseid, transform=image_transforms, rx_type=rx_type, tx_type=tx_type)
   
   #加载数据集
    dataset =FULLmodelDataset(transform=image_transforms)
    print(len(dataset))
    # model = torch.load('model/deep/FusionNet.pth')

    #定义模型
    model=FullModel(num_feature_dim=4)
   
    model.cuda()
    criterion = torch.nn.MSELoss()  # 假设是回归问题
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    #使用交叉验证并评估测试集，选择对应模型名字和存储路径
    cross_validate_with_test(
        dataset, model, criterion, optimizer,
        num_epochs=50, k_folds=3, batch_size=20, test_size=0.25, save_path='model/deep/FusionNettran.pth',model_name='FusionNettran'
    )

    


