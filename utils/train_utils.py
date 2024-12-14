import torch
import yaml
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def setup_logging(log_dir):
    """设置日志记录"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/train.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_optimizer(model, config):
    """获取优化器"""
    return AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        eps=config['optimizer']['eps'],
        weight_decay=config['training']['weight_decay']
    )

def get_scheduler(optimizer, config, steps_per_epoch):
    """获取学习率调度器"""
    return CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'] * steps_per_epoch,
        eta_min=config['scheduler']['min_lr']
    )

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    """保存检查点"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """加载检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_metric']

def accuracy(output, target, topk=(1,)):
    """计算topk准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    fine_top1 = AverageMeter()
    coarse_top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        fine_labels = batch['fine_label'].to(device)
        coarse_labels = batch['coarse_label'].to(device)
        
        # 前向传播
        fine_out, coarse_out = model(images)
        
        # 计算损失
        fine_loss = criterion(fine_out, fine_labels)
        coarse_loss = criterion(coarse_out, coarse_labels)
        loss = fine_loss * 0.7 + coarse_loss * 0.3
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        fine_acc = accuracy(fine_out, fine_labels, topk=(1,))[0]
        coarse_acc = accuracy(coarse_out, coarse_labels, topk=(1,))[0]
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        fine_top1.update(fine_acc.item(), images.size(0))
        coarse_top1.update(coarse_acc.item(), images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'fine_acc': f'{fine_top1.avg:.2f}%',
            'coarse_acc': f'{coarse_top1.avg:.2f}%'
        })
    
    return losses.avg, fine_top1.avg, coarse_top1.avg

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    fine_top1 = AverageMeter()
    coarse_top1 = AverageMeter()
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            fine_labels = batch['fine_label'].to(device)
            coarse_labels = batch['coarse_label'].to(device)
            
            # 前向传播
            fine_out, coarse_out = model(images)
            
            # 计算损失
            fine_loss = criterion(fine_out, fine_labels)
            coarse_loss = criterion(coarse_out, coarse_labels)
            loss = fine_loss * 0.7 + coarse_loss * 0.3
            
            # 计算准确率
            fine_acc = accuracy(fine_out, fine_labels, topk=(1,))[0]
            coarse_acc = accuracy(coarse_out, coarse_labels, topk=(1,))[0]
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            fine_top1.update(fine_acc.item(), images.size(0))
            coarse_top1.update(coarse_acc.item(), images.size(0))
    
    return losses.avg, fine_top1.avg, coarse_top1.avg

def plot_training_progress(train_losses, val_losses, train_accs, val_accs, save_path='training_progress.png'):
    """绘制训练进度图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 