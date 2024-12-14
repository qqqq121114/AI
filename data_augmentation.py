import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

class GroceryDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, is_train=True):
        """
        参数:
            root_dir (str): 数据集根目录
            txt_file (str): 包含图像路径和标签的txt文件
            transform: 数据增强转换
            is_train (bool): 是否是训练模式
        """
        self.root_dir = root_dir
        self.is_train = is_train
        
        # 读取图像路径和标签
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            parts = line.strip().replace(',', '').split()
            if len(parts) >= 3:
                img_path, fine_label, coarse_label = parts
                # 修正路径：将dataset添加到路径中
                full_path = os.path.join(root_dir, 'dataset', img_path)
                if os.path.exists(full_path):
                    self.data.append({
                        'image_path': full_path,
                        'fine_label': int(fine_label),
                        'coarse_label': int(coarse_label)
                    })
                else:
                    print(f"Warning: 找不到图像文件: {full_path}")
        
        print(f"成功加载 {len(self.data)} 个样本")
        
        # 基础转换
        self.basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 训练时的数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ColorJitter(brightness=0.2,    # 颜色抖动
                                 contrast=0.2,
                                 saturation=0.2,
                                 hue=0.1),
            transforms.RandomAffine(degrees=10,       # 仿射变换
                                  translate=(0.1, 0.1),
                                  scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # 随机高斯模糊
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 使用自定义transform或默认transform
        self.transform = transform if transform is not None else \
                        (self.train_transform if is_train else self.basic_transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # 读取图像
            image = Image.open(item['image_path']).convert('RGB')
            
            # 应用转换
            image = self.transform(image)
            
            return {
                'image': image,
                'fine_label': item['fine_label'],
                'coarse_label': item['coarse_label']
            }
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {str(e)}")
            # 返回数据集中的第一个图像作为替代
            return self.__getitem__(0)

def get_data_loaders(root_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器
    """
    # 训练集
    train_dataset = GroceryDataset(
        root_dir=root_dir,
        txt_file=os.path.join(root_dir, 'dataset', 'train.txt'),
        is_train=True
    )
    
    # 验证集
    val_dataset = GroceryDataset(
        root_dir=root_dir,
        txt_file=os.path.join(root_dir, 'dataset', 'val.txt'),
        is_train=False
    )
    
    # 测试集
    test_dataset = GroceryDataset(
        root_dir=root_dir,
        txt_file=os.path.join(root_dir, 'dataset', 'test.txt'),
        is_train=False
    )
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train_dataset)}个样本")
    print(f"验证集: {len(val_dataset)}个样本")
    print(f"测试集: {len(test_dataset)}个样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# 测试代码
if __name__ == '__main__':
    # 测试数据集加载
    root_dir = 'GroceryStoreDataset'
    train_loader, val_loader, test_loader = get_data_loaders(root_dir, batch_size=4)
    
    # 获取一个批次的数据并打印信息
    for batch in train_loader:
        images = batch['image']
        fine_labels = batch['fine_label']
        coarse_labels = batch['coarse_label']
        
        print(f"\n批次信息:")
        print(f"图像形状: {images.shape}")
        print(f"细粒度标签: {fine_labels}")
        print(f"粗粒度标签: {coarse_labels}")
        break 