"""
测试结果解析工具

本脚本用于解析训练日志文件，计算多次实验（不同种子）的统计结果。
支持计算均值、标准差和 95% 置信区间。

主要功能：
    1. 从 log.txt 文件中提取指定指标
    2. 计算多次运行的统计量（均值 ± 标准差）
    3. 支持多实验目录的批量处理

使用示例：
    # 单实验目录
    python parse_test_res.py output/my_experiment
    
    # 多实验目录
    python parse_test_res.py output/my_experiment --multi-exp
    
    # 使用 95% 置信区间
    python parse_test_res.py output/my_experiment --ci95

目录结构假设：
    # 单实验
    my_experiment/
        seed1/log.txt
        seed2/log.txt
        seed3/log.txt
    
    # 多实验
    my_experiment/
        exp-1/seed1/log.txt
        exp-1/seed2/log.txt
        exp-2/seed1/log.txt
        ...

日志格式要求：
    - 包含结束信号（如 "Finish training"）
    - 指标格式：* accuracy: 85.50%
"""

import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dass.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    """
    计算 95% 置信区间的半宽度
    
    公式: CI_95 = 1.96 * std / sqrt(n)
    
    参数：
        res: 结果列表
    
    返回：
        95% 置信区间的半宽度
    """
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    """
    解析单个实验目录
    
    遍历目录下所有子目录的 log.txt 文件，提取指定指标。
    
    参数：
        *metrics: 要提取的指标列表，每个指标是一个字典：
                  {"name": 指标名, "regex": 正则表达式对象}
        directory: 实验目录路径
        args: 命令行参数
        end_signal: 结束信号字符串（表示训练完成）
    
    返回：
        output_results: 指标名到平均值的有序字典
    
    处理逻辑：
        1. 遍历每个种子目录的 log.txt
        2. 等待结束信号出现后才开始提取指标
        3. 收集所有种子的结果
        4. 计算并输出统计量
    """
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False  # 是否遇到结束信号
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                # 检查结束信号
                if line == end_signal:
                    good_to_go = True

                # 提取指标
                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    # 整理结果
    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    # 计算统计量
    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, end_signal):
    """
    主函数
    
    根据模式处理单个或多个实验目录。
    
    参数：
        args: 命令行参数
        end_signal: 结束信号字符串
    """
    # 构建指标正则表达式
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        # 多实验模式：遍历子目录
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        # 输出所有实验的平均结果
        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        # 单实验模式
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="实验结果目录")
    parser.add_argument("--ci95", action="store_true", 
                        help="使用 95% 置信区间代替标准差")
    parser.add_argument("--test-log", action="store_true", 
                        help="解析测试日志（而非训练日志）")
    parser.add_argument("--multi-exp", action="store_true", 
                        help="解析多个实验目录")
    parser.add_argument("--keyword", default="accuracy", type=str, 
                        help="要提取的指标名称")
    args = parser.parse_args()

    # 设置结束信号
    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    main(args, end_signal)
