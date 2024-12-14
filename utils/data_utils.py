import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple, Optional
import numpy as np

class ImageDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._make_dataset()
        
    def _make_dataset(self) -> List[Tuple[str, int]]:
        """创建数据集样本列表"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, filename)
                    samples.append((path, self.class_to_idx[class_name]))
                    
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(config: dict, is_training: bool = True) -> transforms.Compose:
    """获取数据转换"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(config['model']['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment() if config['training']['auto_augment'] else transforms.RandomAffine(0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=config['training']['random_erase']) 
            if config['training']['random_erase'] > 0 else transforms.Lambda(lambda x: x)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(config['model']['img_size'] * 1.14)),
            transforms.CenterCrop(config['model']['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_dataset = ImageDataset(config['data']['train_dir'], 
                               transform=train_transform,
                               is_training=True)
    
    val_dataset = ImageDataset(config['data']['val_dir'], 
                             transform=val_transform,
                             is_training=False)
    
    test_dataset = ImageDataset(config['data']['test_dir'], 
                              transform=val_transform,
                              is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader

def mixup_data(x: torch.Tensor, 
               y: torch.Tensor, 
               alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x: torch.Tensor, 
                y: torch.Tensor, 
                alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    """生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2 