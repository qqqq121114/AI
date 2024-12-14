import torch
import yaml
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    """计算top-k准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res 