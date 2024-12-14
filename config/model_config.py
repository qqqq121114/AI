class ModelConfig:
    # 数据集配置
    image_size = 224
    num_fine_classes = 81
    num_coarse_classes = 43
    channels = 3
    
    # ViT配置
    patch_size = 16
    dim = 768
    depth = 12
    heads = 12
    mlp_dim = 3072
    dropout = 0.1
    emb_dropout = 0.1
    
    # 训练配置
    batch_size = 32
    num_epochs = 100
    learning_rate = 3e-4
    weight_decay = 0.01
    
    # 数据增强配置
    aug_scale = (0.8, 1.0)
    aug_ratio = 0.5
    
    # 损失函数权重
    fine_loss_weight = 0.7
    coarse_loss_weight = 0.3
    
    # 设备配置
    device = 'cuda'  # 或 'cpu'
    num_workers = 4 