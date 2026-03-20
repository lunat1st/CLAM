import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

"""
Generates the 4-panel figure:
  1. Class Weights over epochs        (top-left)  -- from weights_<exp_type>.csv
  2. Eval Accuracy per class          (top-right) -- from <exp_type>.csv
  3. Eval Acc difference (ours-normal)(bottom-left)
  4. Final Eval Accuracy Gap bar chart(bottom-right)

Usage:
    python plot_results.py \
        --ours_csv   cifar10_CLAM_loss_cropbound1.0.csv \
        --normal_csv cifar10_cropbound1.0_l2w0.001.csv \
        --weights_csv weights_cifar10_CLAM_loss_cropbound1.0.csv \
        --task cifar10 \
        --num_classes 5        # set to 10 for full cifar10, 100 for cifar100
"""

def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df

def get_class_cols(df, num_classes):
    """Return the per-class column names (skip 'epoch', 'average', 'valid_acc', 'train_acc')."""
    skip = {'epoch', 'average', 'valid_acc', 'train_acc'}
    cols = [c for c in df.columns if c not in skip]
    return cols[:num_classes]

def plot_results(ours_csv, normal_csv, weights_csv, task, num_classes, output_path):
    ours_df   = load_csv(ours_csv)
    normal_df = load_csv(normal_csv)

    class_cols = get_class_cols(ours_df, num_classes)
    epochs = ours_df['epoch'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors

    # ── Top-left: Class Weights ───────────────────────────────────────────────
    ax = axes[0, 0]
    if weights_csv and os.path.exists(weights_csv):
        weights_df = load_csv(weights_csv)
        weight_cols = get_class_cols(weights_df, num_classes)
        w_epochs = weights_df['epoch'].values
        for i, col in enumerate(weight_cols):
            ax.plot(w_epochs, weights_df[col].values, color=colors[i], label=f'class{i+1}')
    else:
        ax.text(0.5, 0.5, 'weights CSV not found', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'Class Weights (Our Method in {task.upper()})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('class weight')
    ax.legend(fontsize=8)

    # ── Top-right: Eval Accuracy per class (ours) ────────────────────────────
    ax = axes[0, 1]
    for i, col in enumerate(class_cols):
        ax.plot(epochs, ours_df[col].values * 100, color=colors[i], label=f'class{i+1}')
    ax.set_title(f'Evaluation Accuracy (Our Method in {task.upper()})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('eval acc in %')
    ax.legend(fontsize=8)

    # ── Bottom-left: Ours minus Normal per class over epochs ─────────────────
    ax = axes[1, 0]
    normal_class_cols = get_class_cols(normal_df, num_classes)
    # align epochs in case runs have different lengths
    min_len = min(len(ours_df), len(normal_df))
    for i, (our_col, norm_col) in enumerate(zip(class_cols, normal_class_cols)):
        diff = (ours_df[our_col].values[:min_len] - normal_df[norm_col].values[:min_len]) * 100
        ax.plot(epochs[:min_len], diff, color=colors[i], label=f'class{i+1}')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'Evaluation Accuracy (Ours - Normal in {task.upper()})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('eval acc (ours - normal) in %')
    ax.legend(fontsize=8)

    # ── Bottom-right: Final Accuracy Gap bar chart ───────────────────────────
    ax = axes[1, 1]
    final_gaps = []
    for our_col, norm_col in zip(class_cols, normal_class_cols):
        gap = (ours_df[our_col].values[-1] - normal_df[norm_col].values[-1]) * 100
        final_gaps.append(gap)
    x = np.arange(1, num_classes + 1)
    ax.bar(x, final_gaps, color='steelblue', label='ours-normal')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'Final Evaluation Accuracy Gap in {task.upper()}')
    ax.set_xlabel('class')
    ax.set_ylabel('eval acc (ours - normal) in %')
    ax.legend(fontsize=8)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {output_path}')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours_csv',    type=str, required=True,  help='CSV from CLAM/weighted run, e.g. cifar10_CLAM_loss_cropbound1.0.csv')
    parser.add_argument('--normal_csv',  type=str, required=True,  help='CSV from normal loss run, e.g. cifar10_cropbound1.0_l2w0.001.csv')
    parser.add_argument('--weights_csv', type=str, default=None,   help='weights CSV from CLAM run, e.g. weights_cifar10_CLAM_loss_cropbound1.0.csv')
    parser.add_argument('--task',        type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=5,      help='Number of classes to plot (use 5 to match the paper figure, or 10/100 for full datasets)')
    parser.add_argument('--output',      type=str, default='results.png')
    args = parser.parse_args()

    plot_results(
        ours_csv=args.ours_csv,
        normal_csv=args.normal_csv,
        weights_csv=args.weights_csv,
        task=args.task,
        num_classes=args.num_classes,
        output_path=args.output,
    )


# python plot_results.py \
# --ours_csv   cifar10_CLAM_loss_cropbound1.0.csv \
# --normal_csv cifar10_cropbound1.0_l2w0.001.csv \
# --weights_csv weights_cifar10_CLAM_loss_cropbound1.0.csv \
# --task cifar10 \
# --num_classes 5 \
# --output results.png

# python plot.py --ours_csv cifar10_CLAM_loss_cropbound1.0.csv --normal_csv cifar10_cropbound1.0_l2w0.001_copy.csv --weights_csv weights_cifar10_CLAM_loss_cropbound1.0.csv --task cifar1- --num_classes 5 --output results.png