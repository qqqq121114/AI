import pandas as pd
import os
from collections import Counter

# 读取类别信息
def analyze_classes():
    classes_df = pd.read_csv('GroceryStoreDataset/dataset/classes.csv')
    print("\n=== 类别信息 ===")
    print(f"总类别数: {len(classes_df)}")
    print("\n前5个类别示例:")
    print(classes_df.head())
    return classes_df

# 分析数据集分布
def analyze_dataset_split(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fine_labels = []
    coarse_labels = []
    for line in lines:
        # 移除所有逗号，然后分割
        parts = line.replace(',', '').strip().split()
        if len(parts) >= 3:  # 确保行包含足够的信息
            try:
                fine_labels.append(int(parts[-2]))
                coarse_labels.append(int(parts[-1]))
            except ValueError as e:
                print(f"Warning: 无法解析行: {line.strip()}")
                continue
    
    print(f"\n样本示例:")
    print(f"原始行: {lines[0].strip()}")
    print(f"处理后: fine_label={fine_labels[0]}, coarse_label={coarse_labels[0]}")
    
    return {
        'total_samples': len(lines),
        'fine_label_dist': Counter(fine_labels),
        'coarse_label_dist': Counter(coarse_labels)
    }

def main():
    # 1. 分析类别信息
    classes_df = analyze_classes()
    
    # 2. 分析数据集分布
    print("\n=== 数据集分布 ===")
    for split in ['train', 'val', 'test']:
        file_path = f'GroceryStoreDataset/dataset/{split}.txt'
        if os.path.exists(file_path):
            print(f"\n{split}集:")
            try:
                stats = analyze_dataset_split(file_path)
                print(f"样本总数: {stats['total_samples']}")
                print(f"细粒度类别分布: {len(stats['fine_label_dist'])}个类别")
                print(f"粗粒度类别分布: {len(stats['coarse_label_dist'])}个类别")
                
                # 显示每个类别的样本数量
                print("\n细粒度类别样本分布:")
                for label, count in sorted(stats['fine_label_dist'].items()):
                    print(f"类别 {label}: {count}个样本")
            except Exception as e:
                print(f"处理{split}集时出错: {str(e)}")

if __name__ == '__main__':
    main() 