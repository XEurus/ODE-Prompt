import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/ucf101/AdvPT/vit_b16/adv/5_PGD40_16_mix6_sgd_1e3_60/prompt_learner/eval_all_models_clean_test_train_val.csv"
df = pd.read_csv(csv_path)

df = df[df['status'] == 'ok']

epochs = range(5, len(df) * 5 + 1, 5)
if len(df) > 0 and 'model-best.pth.tar' in df['model_file'].values:
    epochs = list(range(5, (len(df) - 2) * 5 + 1, 5)) + [60, 'best']

plt.figure(figsize=(10, 6))
plt.plot(epochs, df['clean_accuracy'], 'b-o', label='Clean', markersize=4)
plt.plot(epochs, df['test_accuracy'], 'r-s', label='Test (PGD)', markersize=4)
plt.plot(epochs, df['train_accuracy'], 'g-^', label='Train', markersize=4)
plt.plot(epochs, df['val_accuracy'], 'm-d', label='Val', markersize=4)

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('UCF101 AdvPT - Clean/Test/Train/Val Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/ucf101/AdvPT/vit_b16/adv/5_PGD40_16_mix6_sgd_1e3_60/prompt_learner/accuracy_curves.png', dpi=150)
plt.show()
print("Saved to accuracy_curves.png")
