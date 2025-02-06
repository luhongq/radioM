import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import os
os.environ['TORCH_HOME'] = '/model/'
class AlexNetDynamic(nn.Module):
    def __init__(self, param_dim=12, input_channels=6):
        super(AlexNetDynamic, self).__init__()

        # self.c00 = nn.Conv2d(in_channels=input_channels, out_channels=24, kernel_size=3, stride=2, padding=1)
        # CNN 部分
        self.c0 = nn.Conv2d(in_channels=input_channels, out_channels=48, kernel_size=5, stride=1, padding=0)
        self.p0 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.c1 = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=4, stride=1, padding=0)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=0)

        # 展平后的维度需要动态计算
        self.flatten_dim = None
        self.fnn_input_dim = None

        # FNN 部分
        self.fc1 = nn.Linear(1, 1)  # 占位，初始化时调整
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 4)
        self.fc_out = nn.Linear(4, 1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x_images, x_sys_params):
        """
        x_images: Tensor of shape [batch_size, channels, height, width]
        x_sys_params: Tensor of shape [batch_size, param_dim]
        """
        # 多张图像拼接到通道维度
        if isinstance(x_images, list):
            x_images = torch.cat(x_images, dim=1)

        # x0 = self.c00(x_images)
        x2 = self.c0(x_images)
        x3 = self.p0(x2)

        x4 = self.c1(x3)
        x5 = self.p1(x4)

        x6 = self.c2(x5)
        x7 = self.c3(x6)

        # 动态计算展平后的维度
        if self.flatten_dim is None:
            self.flatten_dim = x7.view(x7.size(0), -1).size(1)
            self.fnn_input_dim = self.flatten_dim + x_sys_params.size(1)

            # 根据动态计算结果调整全连接层
            self.fc1 = nn.Linear(self.fnn_input_dim, 4096).to(next(self.parameters()).device)

        # 展平 CNN 提取的特征
        x_flat = x7.view(x7.size(0), -1)

        # 结合系统参数

        x_combined = torch.cat((x_flat, x_sys_params), dim=1)

        x = self.relu(self.fc1(x_combined))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc_out(x)

        return x

#
# if __name__ == '__main__':
#     x_image = torch.rand([1, 3, 256, 256])
#     x_sys_params = torch.rand([1, 12])
#
#     model = AlexNetDynamic(param_dim=12)
#     y = model(x_image, x_sys_params)
#
#     print("输出结果:", y)

import os
from osgeo import gdal,osr
import re
from tqdm import tqdm
from rasterio.plot import show
from matplotlib import pyplot as plt
import rasterio
import numpy as np
def extract_coordinates(filename):
    """
    从文件名中提取经纬度信息，如 'ASTGTM_N21E107Q' 提取为 'N21E107'。
    """
    match = re.search(r'([NS]\d+)([EW]\d+)', filename)
    if match:
        return match.group(1) + match.group(2)
    return None

def convert_images_to_geotiff(input_dir, output_dir, epsg_code=4326):
    """
    将一个目录下的图像文件转换为 GeoTIFF 格式，并设置坐标系为指定的 EPSG。

    Args:
        input_dir (str): 输入图像文件目录路径。
        output_dir (str): 输出 GeoTIFF 文件目录路径。
        epsg_code (int): 目标坐标系的 EPSG 代码（默认是 4326）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有图像文件
    for file_name in tqdm(os.listdir(input_dir)):
        # 只处理常见图像文件格式
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff','img')):
            input_path = os.path.join(input_dir, file_name)
            # 提取经纬度信息，构造输出文件名
            coordinates = extract_coordinates(file_name)
            if not coordinates:
                print(f"无法提取坐标信息: {file_name}")
                continue

            output_path = os.path.join(output_dir, f"{coordinates}.tif")


            # # 使用 GDAL 加载图像
            src_ds = gdal.Open(input_path)
            if src_ds is None:
                print(f"无法加载文件: {input_path}")
                continue

                # 获取源影像的投影信息
            src_proj = src_ds.GetProjection()
            src_geotrans = src_ds.GetGeoTransform()

            # 定义目标投影
            dst_proj = osr.SpatialReference()
            dst_proj.ImportFromEPSG(epsg_code)

            # 创建输出影像的投影
            dst_wkt = dst_proj.ExportToWkt()

            # 获取输出影像大小
            dst_x_res = 0.0003057792504332769814  # 输入目标像素宽度（WGS84的经度像素大小）
            dst_y_res = 0.0002632646037167252445  # 输入目标像素高度（WGS84的纬度像素大小）

            # 计算输出范围（边界）
            warp_options = gdal.WarpOptions(
                dstSRS=dst_wkt,
                xRes=dst_x_res,
                yRes=dst_y_res,
                format='GTiff',
                resampleAlg='nearest'
            )

            # 执行重投影
            gdal.Warp(output_path, src_ds, options=warp_options)
            src_ds.FlushCache()
            src_ds = None


            print(f"转换成功: {input_path} -> {output_path}")


##反转灰度值


def convert_images_to_color(input_dir, output_dir, epsg_code=4326):
    """
    将一个目录下的图像文件转换为 GeoTIFF 格式，并设置坐标系为指定的 EPSG。

    Args:
        input_dir (str): 输入图像文件目录路径。
        output_dir (str): 输出 GeoTIFF 文件目录路径。
        epsg_code (int): 目标坐标系的 EPSG 代码（默认是 4326）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有图像文件
    for file_name in tqdm(os.listdir(input_dir)):
        # 只处理常见图像文件格式
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif','img')):
            input_path = os.path.join(input_dir, file_name)
            # 提取经纬度信息，构造输出文件名
            coordinates = extract_coordinates(file_name)
            if not coordinates:
                print(f"无法提取坐标信息: {file_name}")
                continue

            output_path = os.path.join(output_dir, f"{coordinates}.tif")
            # 打开原始影像文件
            with rasterio.open(output_path) as src:
                band = src.read(1)
                print(f"原始数据类型: {band.dtype}, 最大值: {band.max()}, 最小值: {band.min()}")

                # 获取元数据
                meta = src.meta.copy()

                # 处理 nodata 值
                nodata_value = src.nodata if src.nodata else band.min() - 1
                band = np.where(band == nodata_value, nodata_value, band)

                # 获取非 nodata 的最小值和最大值
                valid_mask = band != nodata_value
                min_val, max_val = band[valid_mask].min(), band[valid_mask].max()

                # 反转灰度值：新值 = max_val + min_val - 原值
                reversed_band = np.where(
                    valid_mask,
                    max_val + min_val - band,
                    nodata_value
                ).astype(np.int16)

                # 更新元数据为 Int16 类型
                meta.update({"dtype": "int16", "count": 1, "nodata": nodata_value})

                # 保存处理后的影像到文件
                temp_output_path = output_path.replace(".tif", "_temp.tif")
                with rasterio.open(temp_output_path, "w", **meta) as dst:
                    dst.write(reversed_band, 1)


            print(f"转换成功: {input_path} -> {output_path}")

# if __name__ == "__main__":
#     input_directory = r"G://30mDEM"
#     output_directory = r"F://jsdaima//30DEM//"
#     convert_images_to_geotiff(input_directory, output_directory)
#     # convert_images_to_color(output_directory, output_directory)

import os


def rename_files_in_subdirectories(base_directory):
    """
    遍历一个目录下的所有子目录，将文件名从 masked_x.tiff 修改为 x.tiff。

    Args:
        base_directory (str): 顶级目录路径。
    """
    for subdir, _, files in tqdm(os.walk(base_directory)):
        for file in files:
            if file.startswith("masked_") and file.endswith(".tiff"):
                old_path = os.path.join(subdir, file)
                new_filename = file.replace("masked_", "")
                new_path = os.path.join(subdir, new_filename)

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


# # 示例用法
# # base_directory = "dataset/524287"  # 替换为你的目标目录
# # rename_files_in_subdirectories(base_directory)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# 创建反转的 Turbo 颜色条
# 创建反转的 Turbo 颜色条

def plot_reversed_vertical_turbo_colorbar():
    # 获取 Turbo 颜色映射并反转
    turbo_cmap = plt.cm.get_cmap('turbo')


    # 定义颜色条的范围
    norm = colors.Normalize(vmin=-110, vmax=-40)  # 颜色条范围

    # 创建图形
    fig, ax = plt.subplots(figsize=(4, 8))  # 图形尺寸，确保颜色条清晰
    # 在颜色条右侧增加显示空间
    fig.subplots_adjust(left=0.3, right=0.7)  # 调整图形边距以向右移动颜色条
    # 添加颜色条
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=turbo_cmap),
        ax=ax,
        orientation='vertical',  # 竖直方向
    )

    # 设置标题
    cbar.ax.set_title('SS-RSRP (dBm)', fontsize=14, pad=15)  # 设置标题在颜色条上方

    # 设置刻度和标签
    cbar.set_ticks([-110, -75, -40])  # 设置刻度位置
    cbar.ax.set_yticklabels(['-95', '-75', '-54'], fontsize=12)  # 设置刻度标签

    # 移除多余的图形内容，仅保留颜色条
    ax.remove()

    # 显示图形
    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBranch(nn.Module):
    def __init__(self, pretrained=True, output_dim=128):
        super(ResNetBranch, self).__init__()
        # 使用预训练的 ResNet
        resnet = models.resnet18(pretrained=pretrained)
        # 修改第一层以适配 256x256 彩色图像
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉最后的全连接层
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (batch_size, 512, 1, 1)
        self.fc = nn.Linear(512, output_dim)  # 将 ResNet 提取的特征降维

    def forward(self, x):
        x = self.feature_extractor(x)  # 提取特征
        x = torch.flatten(x, 1)  # 展平为 (batch_size, 512)
        x = self.fc(x)  # 降维为 (batch_size, output_dim)
        return x

class MLPBranch(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[32, 16], output_dim=64):
        super(MLPBranch, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # 防止过拟合
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FullModel(nn.Module):
    def __init__(self, image_feature_dim=128, num_feature_dim=64, final_output_dim=1):
        super(FullModel, self).__init__()
        self.image_branch = ResNetBranch(output_dim=image_feature_dim)
        self.feature_branch = MLPBranch(input_dim=4, output_dim=num_feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + num_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, final_output_dim)
        )

    def forward(self, image, features):
        image_out = self.image_branch(image)  # 图像特征提取
        feature_out = self.feature_branch(features)  # 数值特征提取
        combined = torch.cat([image_out, feature_out], dim=1)  # 拼接特征
        output = self.fc(combined)  # 回归输出
        return output




from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

def diagonal_flip(image):
    """对角线翻转"""
    return image.transpose(Image.TRANSPOSE)
def diagonal_flip(image):
    """对角线翻转"""
    return image.transpose(Image.TRANSPOSE)

def generate_augmented_dataset(baseid, rx_type, output_dir):
    """
    对图像进行数据增强（水平翻转、竖直翻转、对角线翻转）并扩展数据集。

    Args:
        baseid (int): 数据集 ID。
        rx_type (list): Rx 图像类型列表。
        output_dir (str): 增强后的图像和 CSV 文件保存路径。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 加载原始 CSV 文件
    csv_path = f"dataset/{baseid}/{baseid}.csv"
    params_data = pd.read_csv(csv_path)

    # 增强后的 CSV 数据
    augmented_data = []
    current_id = len(params_data) + 1  # 增强数据的 ID 从原始数据的长度加 1 开始

    for idx, row in tqdm(params_data.iterrows(), total=len(params_data)):
        for name_type in rx_type:
            input_img_path = f"dataset/{baseid}/{name_type}/rx/{idx + 1}.tiff"

            if not os.path.exists(input_img_path):
                print(f"Warning: Image not found at {input_img_path}")
                continue

            # 打开原始图像
            image = Image.open(input_img_path)

            # 增强变换列表
            transforms_list = [
                ("horizontal", image.transpose(Image.FLIP_LEFT_RIGHT)),  # 水平翻转
                ("vertical", image.transpose(Image.FLIP_TOP_BOTTOM)),  # 垂直翻转
                ("diagonal", diagonal_flip(image))  # 对角线翻转
            ]

            for transform_name, transformed_image in transforms_list:
                # 保存增强图像
                augmented_img_name = f"{current_id}.tiff"
                augmented_img_path = os.path.join(output_dir, augmented_img_name)
                transformed_image.save(augmented_img_path)
                print(f'图片已存储到{augmented_img_path}')

                # 增强样本的 CSV 数据
                new_row = row.copy()
                new_row['id'] = current_id  # 确保增强样本拥有唯一的 ID
                print(current_id)
                augmented_data.append(new_row)
                current_id += 1

    # 合并原始数据和增强数据
    augmented_df = pd.DataFrame(augmented_data)
    full_data = pd.concat([params_data, augmented_df], ignore_index=True)

    # 保存完整的 CSV 文件
    full_csv_path = os.path.join(f'dataset/{baseid}/', f"{baseid}_full.csv")
    full_data.to_csv(full_csv_path, index=False)
    print(f"Full dataset saved to {full_csv_path}")


# # 示例：对 baseid=751630 的数据集进行扩展
# baseid=524287
# generate_augmented_dataset(
#     baseid=baseid,
#     rx_type=['imgw'],  # Rx 图像类型
#     output_dir=f"dataset/{baseid}/imgw_high/rx/",  # 增强数据保存目录
# )

