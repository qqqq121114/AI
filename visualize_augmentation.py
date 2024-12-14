import torch
import matplotlib.pyplot as plt
from data_augmentation import GroceryDataset
import torchvision.transforms as transforms
import numpy as np

def show_augmented_images(dataset, num_augmentations=5):
    # 选择一个样本
    sample_idx = 0
    original_sample = dataset[sample_idx]
    
    # 创建子图
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 3))
    fig.suptitle('数据增强效果展示')
    
    # 显示原始图像
    img = original_sample['image']
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    axes[0].imshow(img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示增强后的图像
    for i in range(num_augmentations):
        augmented_sample = dataset[sample_idx]
        img = augmented_sample['image']
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'增强 {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 创建数据集实例
    dataset = GroceryDataset(
        root_dir='GroceryStoreDataset',
        txt_file='GroceryStoreDataset/dataset/train.txt',
        is_train=True
    )
    
    # 显示增强效果
    show_augmented_images(dataset) 