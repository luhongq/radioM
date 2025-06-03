import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
import os
import csv
from tqdm import tqdm
from thop import profile
from sklearn.model_selection import KFold, train_test_split
from torchvision import transforms
import torchattacks as ta
from loader import PropagationDataset,train_validate_test_split,FULLmodelDataset1
from copy import deepcopy
# 模型分支
from AlexNet import FullModel,MobileNetTransformerModel,TransformerFullModel,MobileNetFullModel,ImageOnlyModel,NumericOnlyModel  # 原模型
import logging

# 将日志级别设置为 WARNING，屏蔽 INFO 和 DEBUG 信息
logging.basicConfig(level=logging.WARNING)
import random

# ✅ **测试集综合评估函数**
def evaluate_on_testset(model, dataloader, criterion, input_type,device="cuda"):
    """
    在测试集上同时计算：
    - **测试集验证损失 (MSE + RMSE)**
    - **整体预测耗时**
    - **模型复杂度 (FLOPs、参数量、平均推理时间)**
    - **鲁棒性测试 (MAE、MSE)**
    """
    model.eval()

    # 初始化累计变量
    test_mse, test_rmse, total_samples = 0.0, 0.0, 0
    total_macs, total_params, total_time = 0.0, 0.0, 0.0


    with torch.no_grad():
        test_bar=tqdm(dataloader,  desc="测试集综合评估")
        for batch in test_bar:
            images = torch.cat([img.to(device) for img in batch["images"]], dim=1)
            sys_params = batch["sys_params"].to(device)
            labels = batch["label"].to(device)
            batch_size = images.size(0)
            # ✅ 根据 input_type 动态调整输入格式
            if input_type == "both":
                inputs = (images, sys_params)
            elif input_type == "image":
                inputs = images
            elif input_type == "numeric":
                inputs = sys_params
            else:
                raise ValueError(f"❌ 不支持的 input_type: {input_type}")

            # ✅ 计算 FLOPs 和参数量
            if input_type == "both":
                macs, params = profile(model, inputs=(images, sys_params),verbose=False)
            else:
                macs, params = profile(model, inputs=(inputs,),verbose=False)

            total_macs += macs
            total_params += params

            # ✅ 推理时间测量
            start = time.time()

            if input_type == "both":
                outputs = model(images, sys_params)
            else:
                outputs = model(inputs)

            end = time.time()
            infer_time = end - start
            total_time += infer_time

            # ✅ 累计 MSE 和 RMSE
            mse = torch.mean((outputs.squeeze() - labels) ** 2).item()
            rmse = np.sqrt(mse)
            test_bar.set_postfix({'batch test loss:':mse,"batch test rmse:":rmse})

            test_mse += mse * batch_size
            test_rmse += rmse * batch_size
            total_samples += batch_size


        test_bar.close()

    # **计算平均值**
    avg_test_mse = test_mse / total_samples
    avg_test_rmse = test_rmse / total_samples
    avg_macs = total_macs / len(dataloader)
    avg_params = total_params / len(dataloader)
    avg_time = total_time / len(dataloader)


    # **打印结果**
    print(f"\n✅ 测试集评估结果：")
    print(f"📌 测试集验证损失 (MSE): {avg_test_mse:.6f}")
    print(f"📌 测试集验证损失 (RMSE): {avg_test_rmse:.6f}")
    print(f"📌 模型复杂度 -> FLOPs: {avg_macs:.2f}, Params: {avg_params:.2f}, 平均推理时间: {avg_time:.6f} 秒")
    print(f"📌 预测整体耗时: 总时间: {total_time:.4f} 秒, 平均样本推理时间: {avg_time / total_samples:.6f} 秒")


    return avg_test_mse, avg_test_rmse, avg_macs, avg_params, avg_time, total_time, avg_time / total_samples


def add_image_noise(images, noise_level=0.05):
    """对图像数据加噪"""
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, 0, 1)  # 限制像素值在[0,1]之间

def noisy_training_batch(batch, device ='cuda',noise_prob=0, noise_level=0.1):
    """对 batch 中的部分样本加噪,noise_prob=0默认不加噪"""
    images = torch.cat([img.to(device) for img in batch["images"]], dim=1)
    sys_params = batch["sys_params"].to(device)
    labels = batch["label"].to(device)

    # 加噪逻辑：样本加噪
    if random.random() < noise_prob:
        images = add_image_noise(images, noise_level=noise_level)


    return images, sys_params, labels


def cross_validate(dataset, models_to_test, criterion, optimizer, num_epochs=50, k_folds=3, batch_size=32, test_size=0.2,
                   device="cuda"):
    """
    一次性完成：
    - **所有模型的交叉验证**
    - **在测试集上同时计算所有指标**
    - **自动保存结果**
    """
    # **划分数据集**
    train_val_indices, test_indices = train_validate_test_split(dataset, test_size=test_size)
    train_val_dataset = Subset(dataset, train_val_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f'dataset;{len(dataset)},train:{len(train_val_dataset)},test;{len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # **初始化 CSV 文件**
    results_file = "results_sim_radioM.csv"
    processed_models = set()

    # ✅ 新增：读取已存在的模型记录

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Test MSE", "Test RMSE", "FLOPs", "Params", "Infer Time",
                "Total Time", "Avg Sample Time"
            ])
    else:
        with open(results_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                if row:  # 避免空行
                    processed_models.add(row[0])

    for model_name, config in models_to_test.items():
        model = config["model"]
        input_type = config["input_type"]
        best_model_path = f"model/deep/{model_name}_best.pth"
        if model_name in processed_models:
            print(f"\n⏩ 跳过已完成的模型: {model_name}")
            continue
        print(f"\n🔹 训练模型: {model_name}")
        #
        optimizer1 = optimizer(model.parameters(), lr=1e-3)
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_val_loss = float("inf")


        model.to(device)
        avg_val_loss,avg_train_loss,train_rmse,val_rmse=0,0,0,0
        # ✅ 修改：基于原始训练验证索引进行K折拆分
        for fold, (train_sub_idx, val_sub_idx) in enumerate(kfold.split(train_val_indices)):
            print(f"\n🔹 Fold {fold + 1}/{k_folds}")
            # 获取实际索引
            train_indices = train_val_indices[train_sub_idx]
            val_indices = train_val_indices[val_sub_idx]
            # 创建Subset
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            # 创建DataLoader
            train_loader = DataLoader(train_subset, batch_size=20, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=20, shuffle=True)
            # ✅ 早停机制变量

            epochs_ubar=tqdm(range(num_epochs), desc="Training Epochs")
            for epoch in epochs_ubar:
                epochs_ubar.set_postfix({"Model": model_name,
                                    "Fold": f"{fold + 1}",
                                    "Epoch": f"{epoch + 1}/{num_epochs}",
                                    "Train Loss": f"{avg_train_loss:.4f}",
                                    "Train RMSE": f"{train_rmse:.4f}",
                                    "Val Loss": f"{avg_val_loss:.4f}",
                                    "Val RMSE": f"{val_rmse:.4f}"
                                         })
                model.train()
                train_loss = 0.0
                train_ubar=tqdm(train_loader, desc="Training Batches",leave=False)
                for batch in train_ubar:
                    images, sys_params, labels = noisy_training_batch(batch, noise_prob=0, noise_level=0.1)

                    # ✅ 动态处理输入类型
                    if input_type == "both":

                        outputs = model(images, sys_params)
                    elif input_type == "image":

                        outputs = model(images)
                    elif input_type == "numeric":

                        outputs = model(sys_params)
                    labels = batch["label"].to(device)

                    loss = criterion(outputs.squeeze(), labels)
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()
                    train_loss += loss.item()
                    train_ubar.set_postfix({'batch train loss:':loss.item(),"batch train rmse:":loss.item()**0.5})

                avg_train_loss = train_loss / len(train_loader)
                train_rmse =avg_train_loss ** 0.5

                train_ubar.close()

                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_rmse_sum = 0.0  # RMSE 累加器
                with torch.no_grad():
                    val_bar= tqdm(val_loader, desc="Val Batches",leave=False)
                    for batch in val_bar:
                        # ✅ 动态处理输入类型
                        if input_type == "both":
                            images = torch.cat([img.to(device) for img in batch["images"]], dim=1)
                            sys_params = batch["sys_params"].to(device)
                            outputs = model(images, sys_params)
                        elif input_type == "image":
                            images = torch.cat([img.to(device) for img in batch["images"]], dim=1)
                            outputs = model(images)
                        elif input_type == "numeric":
                            sys_params = batch["sys_params"].to(device)
                            outputs = model(sys_params)
                        labels = batch["label"].to(device)


                        loss = criterion(outputs.squeeze(), labels)
                        val_loss += loss.item()
                        # 计算 RMSE 的累加部分
                        val_rmse_sum += ((outputs.squeeze() - labels) ** 2).sum().item()
                        val_bar.set_postfix({'batch val loss:': loss.item(), "batch val rmse:": loss.item() ** 0.5})
                    val_bar.close()


                avg_val_loss = val_loss / len(val_loader)
                # 计算验证集的 RMSE
                val_rmse = avg_val_loss ** 0.5
                # 保存最佳模型
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    torch.save(model, best_model_path)




        # ✅ **在集上评估**
        model=torch.load(best_model_path,weights_only=False)
        model.to(device)

        # ✅ **在测试集上计算所有指标**
        avg_test_mse, avg_test_rmse, avg_macs, avg_params, avg_time, total_time, avg_sample_time = evaluate_on_testset(
            model, test_loader, criterion, input_type=input_type
        )

        # ✅ **保存结果**
        with open(results_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                model_name, avg_test_mse, avg_test_rmse, avg_macs, avg_params, avg_time,
                total_time, avg_sample_time
            ])
            f.flush()  # 确保立即写入磁盘

        tqdm.write(f"✅ 结果已保存到 {results_file}`")


if __name__ == '__main__':

    # 一次性运行所有模型
    models_to_test = {
        "ResNet_MLP": {"model": FullModel(num_feature_dim=4), "input_type": "both"},
        "ImageOnly": {"model": ImageOnlyModel(), "input_type": "image"},       # 仅图像分支
        "NumericOnly": {"model": NumericOnlyModel(), "input_type": "numeric"}, # 仅数值分支
        "MobileNet_MLP": {"model": MobileNetFullModel(num_feature_dim=4), "input_type": "both"},
        "ResNet_Transformer": {"model": TransformerFullModel(num_feature_dim=4), "input_type": "both"},
        "MobileNet_Transformer": {"model": MobileNetTransformerModel(num_feature_dim=4), "input_type": "both"}
    }

    # 数据预处理 - 图像标准化
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化
        transforms.Lambda(lambda x: x.to(torch.float32))

    ])
    # dataset = PropagationDataset(baseids=[751630, 524287], transform=image_transforms, rx_type=['imgw'],tx_type=[])
    dataset =FULLmodelDataset1(transform=image_transforms)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam

    # 一次性训练
    print("CUDA Available:", torch.cuda.is_available())
    print(torch.version.cuda)  # 输出PyTorch编译时的CUDA版本
    print("Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))

    torch.backends.cudnn.benchmark = True  # 启用CUDA加速算法
    cross_validate(dataset, models_to_test, criterion, optimizer,device="cuda" if torch.cuda.is_available() else "cpu")
