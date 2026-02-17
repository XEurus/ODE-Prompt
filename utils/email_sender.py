"""
邮件发送工具 - 用于发送训练结果到指定邮箱
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import argparse
from datetime import datetime
import re
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================== 邮箱配置 ====================
# 请在此处填写邮箱密码
EMAIL_PASSWORD = "KAdPRentpwivX4Tc"  # TODO: 请填写 smtp_sending@126.com 的密码

SMTP_CONFIG = {
    "smtp_host": "smtp.126.com",
    "smtp_port": 25,
    "username": "smtp_sending@126.com",
    "from_addr": "smtp_sending@126.com",
    "to_addrs": ["eurus_receive@163.com"],
    "use_tls": True,
}
# =================================================


def send_email(
    subject: str,
    body: str,
    attachments: list = None,
    body_type: str = "plain"
):
    """
    发送邮件

    Args:
        subject: 邮件主题
        body: 邮件正文
        attachments: 附件路径列表，可选
        body_type: 正文类型，'plain' 或 'html'

    Returns:
        bool: 发送是否成功
    """
    if not EMAIL_PASSWORD:
        raise ValueError("请先在 EMAIL_PASSWORD 变量中填写邮箱密码")

    config = SMTP_CONFIG
    msg = MIMEMultipart()
    msg['From'] = config['from_addr']
    msg['To'] = ', '.join(config['to_addrs'])
    msg['Subject'] = subject

    # 添加正文
    msg.attach(MIMEText(body, body_type, 'utf-8'))

    # 添加附件
    if attachments:
        for file_path in attachments:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(file_path)}'
                )
                msg.attach(part)
            else:
                print(f"警告: 附件不存在: {file_path}")

    try:
        server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
        
        if config['use_tls']:
            server.starttls()
        
        server.login(config['username'], EMAIL_PASSWORD)
        server.sendmail(config['from_addr'], config['to_addrs'], msg.as_string())
        server.quit()
        
        print(f"邮件发送成功: {subject}")
        return True
        
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False


def send_training_results(
    experiment_name: str,
    results_text: str,
    result_files: list = None,
    additional_info: dict = None
):
    """
    发送训练结果邮件

    Args:
        experiment_name: 实验名称
        results_text: 训练结果文本
        result_files: 结果文件路径列表（如图表、日志等）
        additional_info: 额外信息字典

    Returns:
        bool: 发送是否成功
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    subject = f"[训练完成] {experiment_name} - {timestamp}"
    
    # 构建邮件正文
    body_lines = [
        f"实验名称: {experiment_name}",
        f"完成时间: {timestamp}",
        "",
        "=" * 40,
        "训练结果:",
        "=" * 40,
        "",
        results_text,
    ]
    
    if additional_info:
        body_lines.extend(["", "=" * 40, "额外信息:", "=" * 40, ""])
        for key, value in additional_info.items():
            body_lines.append(f"{key}: {value}")
    
    body = "\n".join(body_lines)
    
    return send_email(subject, body, attachments=result_files)


def _safe_float(token: str):
    token = str(token).strip()
    if token in {"NA", "None", "nan", "NaN", "-"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def read_last_lines(file_path: str, n: int = 50) -> str:
    """读取文件最后 n 行。"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = deque(f, maxlen=n)
    return "".join(lines)


def parse_epoch_metrics(log_file: str):
    """
    解析训练日志中的结构化 Epoch 指标。

    期望格式：
    [EpochMetrics] epoch=1 train_adv_acc=... val_adv_acc=... test_partial_adv_acc=...
    """
    pattern = re.compile(
        r"\[EpochMetrics\]\s+epoch=(\d+)\s+"
        r"train_adv_acc=([^\s]+)\s+"
        r"val_adv_acc=([^\s]+)\s+"
        r"test_partial_adv_acc=([^\s]+)"
    )

    epochs, train_vals, val_vals, test_vals = [], [], [], []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epochs.append(int(match.group(1)))
            train_vals.append(_safe_float(match.group(2)))
            val_vals.append(_safe_float(match.group(3)))
            test_vals.append(_safe_float(match.group(4)))

    return {
        "epochs": epochs,
        "train_adv_acc": train_vals,
        "val_adv_acc": val_vals,
        "test_partial_adv_acc": test_vals,
    }


def parse_batch_train_acc(log_file: str):
    """兼容旧日志：从 batch 日志中提取平均训练准确率 acc (...)."""
    acc_pattern = re.compile(r"acc\s+[\d\.]+\s+\(([\d\.]+)\)")
    accs = []
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = acc_pattern.search(line)
            if match:
                accs.append(float(match.group(1)))
    return accs


def generate_accuracy_curve(log_file: str, output_file: str = None):
    """
    根据日志生成训练准确率曲线图。

    优先使用 [EpochMetrics]（train/val/test 三条曲线）；
    若不存在则回退到 batch 级训练 acc 曲线。
    """
    if output_file is None:
        output_file = f"{log_file}.accuracy_curve.png"

    metrics = parse_epoch_metrics(log_file)
    has_epoch_metrics = len(metrics["epochs"]) > 0

    plt.figure(figsize=(12, 6))

    if has_epoch_metrics:
        epochs = metrics["epochs"]

        def _plot_series(values, label):
            xs = [e for e, v in zip(epochs, values) if v is not None]
            ys = [v for v in values if v is not None]
            if xs:
                plt.plot(xs, ys, marker='o', label=label)

        _plot_series(metrics["train_adv_acc"], "Train Adv Acc")
        _plot_series(metrics["val_adv_acc"], "Val Adv Acc")
        _plot_series(metrics["test_partial_adv_acc"], "Test Partial Adv Acc")

        plt.xlabel("Epoch")
        plt.title("Adversarial Accuracy Curve (Epoch-level)")
    else:
        train_accs = parse_batch_train_acc(log_file)
        if not train_accs:
            raise ValueError("日志中未找到可用于绘图的准确率数据")
        steps = list(range(1, len(train_accs) + 1))
        plt.plot(steps, train_accs, label="Train Acc (running avg)")
        plt.xlabel("Logged Steps")
        plt.title("Training Accuracy Curve (Batch-level)")
        metrics = {
            "epochs": [len(train_accs)],
            "train_adv_acc": [train_accs[-1]],
            "val_adv_acc": [None],
            "test_partial_adv_acc": [None],
        }

    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file, metrics


# ==================== 使用示例 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="发送训练结果邮件")
    parser.add_argument("--exp_name", "-e", type=str, required=True, help="实验名称")
    parser.add_argument("--log_file", "-l", type=str, help="日志文件路径，正文显示最后50行，附件发送完整文件+准确率曲线图")
    parser.add_argument("--result_file", "-r", type=str, help="结果文件路径（包含训练结果的文本文件）")
    parser.add_argument("--result_text", "-t", type=str, help="直接传入的结果文本")
    parser.add_argument("--attachments", "-a", type=str, help="额外附件文件路径，多个文件用逗号分隔")
    
    args = parser.parse_args()
    
    # 获取结果文本（邮件正文）
    results_text = ""
    attachment_list = []
    additional_info = None
    
    # 如果指定了日志文件，读取最后30行作为正文，并将完整日志作为附件
    if args.log_file:
        if os.path.exists(args.log_file):
            # 读取最后50行作为正文
            results_text = read_last_lines(args.log_file, n=50)
            # 将完整日志作为附件
            attachment_list.append(args.log_file)

            # 基于日志生成准确率曲线图并附带最后一轮指标
            try:
                curve_file, parsed_metrics = generate_accuracy_curve(args.log_file)
                attachment_list.append(curve_file)

                if parsed_metrics["epochs"]:
                    idx = -1
                    additional_info = {
                        "final_epoch": parsed_metrics["epochs"][idx],
                        "final_train_adv_acc": parsed_metrics["train_adv_acc"][idx],
                        "final_val_adv_acc": parsed_metrics["val_adv_acc"][idx],
                        "final_test_partial_adv_acc": parsed_metrics["test_partial_adv_acc"][idx],
                        "accuracy_curve": curve_file,
                    }
            except Exception as e:
                print(f"警告: 生成准确率曲线失败: {e}")
        else:
            print(f"错误: 日志文件不存在: {args.log_file}")
            exit(1)
    elif args.result_file:
        if os.path.exists(args.result_file):
            with open(args.result_file, 'r', encoding='utf-8') as f:
                results_text = f.read()
        else:
            print(f"错误: 结果文件不存在: {args.result_file}")
            exit(1)
    elif args.result_text:
        results_text = args.result_text
    else:
        print("错误: 请提供 --log_file、--result_file 或 --result_text 参数")
        exit(1)
    
    # 添加额外附件
    if args.attachments:
        extra_attachments = [f.strip() for f in args.attachments.split(",") if f.strip()]
        attachment_list.extend(extra_attachments)

    # 去重（保持原顺序）
    attachment_list = list(dict.fromkeys(attachment_list))
    
    # 发送邮件
    success = send_training_results(
        experiment_name=args.exp_name,
        results_text=results_text,
        result_files=attachment_list,
        additional_info=additional_info,
    )
    
    exit(0 if success else 1)
