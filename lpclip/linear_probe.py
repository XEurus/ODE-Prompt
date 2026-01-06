"""
CLIP 线性探针 (Linear Probe)

本脚本实现了基于 CLIP 特征的线性分类器训练和评估。
使用逻辑回归作为线性分类头，通过二分搜索自动调优正则化参数。

线性探针方法：
    1. 使用预训练 CLIP 提取图像特征（固定）
    2. 在特征上训练一个线性分类器
    3. 通过验证集调优超参数

主要特点：
    - 支持 Few-shot 学习（1, 2, 4, 8, 16 shot）
    - 自动超参数搜索（L2 正则化系数）
    - 多次运行取平均（减少随机性影响）
    - 二分搜索优化正则化强度

使用方法：
    python linear_probe.py \
        --dataset OxfordPets \
        --feature_dir clip_feat \
        --num_step 8 \
        --num_run 10

输入文件（由 feat_extractor.py 生成）：
    {feature_dir}/{dataset}/train.npz
    {feature_dir}/{dataset}/val.npz
    {feature_dir}/{dataset}/test.npz

输出文件：
    report/{feature_dir}_s{num_step}r{num_run}.txt         # 汇总结果
    report/{feature_dir}_s{num_step}r{num_run}_details.txt # 详细结果
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="", help="数据集名称")
parser.add_argument("--num_step", type=int, default=8, help="二分搜索步数")
parser.add_argument("--num_run", type=int, default=10, help="运行次数（不同随机种子）")
parser.add_argument("--feature_dir", type=str, default="clip_feat", help="特征目录")
args = parser.parse_args()

# 加载数据集特征
dataset = args.dataset
dataset_path = os.path.join(f"{args.feature_dir}", dataset)

# 加载训练、验证、测试特征
train_file = np.load(os.path.join(dataset_path, "train.npz"))
train_feature, train_label = train_file["feature_list"], train_file["label_list"]
val_file = np.load(os.path.join(dataset_path, "val.npz"))
val_feature, val_label = val_file["feature_list"], val_file["label_list"]
test_file = np.load(os.path.join(dataset_path, "test.npz"))
test_feature, test_label = test_file["feature_list"], test_file["label_list"]

# 创建报告目录
os.makedirs("report", exist_ok=True)

# 验证集采样数量映射（不同 shot 数使用不同的验证集大小）
val_shot_list = {1: 1, 2: 2, 4: 4, 8: 4, 16: 4}

# 遍历不同的 Few-shot 设置
for num_shot in [1, 2, 4, 8, 16]:
    # 记录每次运行每步的测试准确率
    test_acc_step_list = np.zeros([args.num_run, args.num_step])
    
    for seed in range(1, args.num_run + 1):
        np.random.seed(seed)
        print(f"-- Seed: {seed} --------------------------------------------------------------")
        
        # ========================================
        # Few-shot 采样
        # ========================================
        # 从训练集中为每个类别采样 num_shot 个样本
        all_label_list = np.unique(train_label)
        selected_idx_list = []
        for label in all_label_list:
            label_collection = np.where(train_label == label)[0]
            selected_idx = np.random.choice(label_collection, size=num_shot, replace=False)
            selected_idx_list.extend(selected_idx)

        fewshot_train_feature = train_feature[selected_idx_list]
        fewshot_train_label = train_label[selected_idx_list]

        # 验证集采样
        val_num_shot = val_shot_list[num_shot]
        val_selected_idx_list = []
        for label in all_label_list:
            label_collection = np.where(val_label == label)[0]
            selected_idx = np.random.choice(label_collection, size=val_num_shot, replace=False)
            val_selected_idx_list.extend(selected_idx)

        fewshot_val_feature = val_feature[val_selected_idx_list]
        fewshot_val_label = val_label[val_selected_idx_list]

        # ========================================
        # 初始搜索：粗粒度网格搜索
        # ========================================
        # 在对数空间搜索最佳正则化系数
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            # 训练逻辑回归分类器
            clf = LogisticRegression(
                solver="lbfgs",      # L-BFGS 优化器
                max_iter=1000,       # 最大迭代次数
                penalty="l2",        # L2 正则化
                C=c_weight           # 正则化系数的倒数
            ).fit(fewshot_train_feature, fewshot_train_label)
            
            # 验证集评估
            pred = clf.predict(fewshot_val_feature)
            acc_val = sum(pred == fewshot_val_label) / len(fewshot_val_label)
            acc_list.append(acc_val)

        print(acc_list, flush=True)

        # ========================================
        # 二分搜索：精细调优
        # ========================================
        # 找到最佳 C 值附近的搜索范围
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak  # 搜索范围：0.1x 到 10x

        def binary_search(c_left, c_right, seed, step, test_acc_step_list):
            """
            二分搜索正则化系数
            
            在 [c_left, c_right] 范围内搜索最佳 C 值。
            比较两个端点的验证准确率，保留较好的一侧。
            
            参数：
                c_left: 搜索范围左边界
                c_right: 搜索范围右边界
                seed: 当前随机种子
                step: 当前搜索步数
                test_acc_step_list: 测试准确率记录数组
            
            返回：
                更新后的搜索范围和记录数组
            """
            # 训练左边界分类器
            clf_left = LogisticRegression(
                solver="lbfgs", max_iter=1000, penalty="l2", C=c_left
            ).fit(fewshot_train_feature, fewshot_train_label)
            pred_left = clf_left.predict(fewshot_val_feature)
            acc_left = sum(pred_left == fewshot_val_label) / len(fewshot_val_label)
            print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

            # 训练右边界分类器
            clf_right = LogisticRegression(
                solver="lbfgs", max_iter=1000, penalty="l2", C=c_right
            ).fit(fewshot_train_feature, fewshot_train_label)
            pred_right = clf_right.predict(fewshot_val_feature)
            acc_right = sum(pred_right == fewshot_val_label) / len(fewshot_val_label)
            print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

            # 选择较好的一侧，并缩小搜索范围
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # 下一步搜索范围：中点到右边界
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # 下一步搜索范围：左边界到中点
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            # 在测试集上评估
            pred = clf_final.predict(test_feature)
            test_acc = 100 * sum(pred == test_label) / len(pred)
            print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
            test_acc_step_list[seed - 1, step] = test_acc

            # 保存详细结果
            saveline = "{}, seed {}, {} shot, weight {}, test_acc {:.2f}\n".format(
                dataset, seed, num_shot, c_final, test_acc)
            with open(
                "./report/{}_s{}r{}_details.txt".format(args.feature_dir, args.num_step, args.num_run),
                "a+"
            ) as writer:
                writer.write(saveline)
            
            return (
                np.power(10, c_left),
                np.power(10, c_right),
                seed,
                step,
                test_acc_step_list,
            )

        # 执行多步二分搜索
        for step in range(args.num_step):
            print(
                f"{dataset}, {num_shot} Shot, Round {step}: {c_left}/{c_right}",
                flush=True,
            )
            c_left, c_right, seed, step, test_acc_step_list = binary_search(
                c_left, c_right, seed, step, test_acc_step_list)
    
    # ========================================
    # 汇总结果
    # ========================================
    # 使用最后一步的结果计算统计量
    test_acc_list = test_acc_step_list[:, -1]
    acc_mean = np.mean(test_acc_list)
    acc_std = np.std(test_acc_list)
    save_line = "{}, {} Shot, Test acc stat: {:.2f} ({:.2f})\n".format(
        dataset, num_shot, acc_mean, acc_std)
    print(save_line, flush=True)
    
    # 保存汇总结果
    with open(
        "./report/{}_s{}r{}.txt".format(args.feature_dir, args.num_step, args.num_run),
        "a+"
    ) as writer:
        writer.write(save_line)
