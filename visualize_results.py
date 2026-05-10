import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")


def load_all_logs():
    """
    加载并整合所有训练日志
    """
    all_dfs = []

    # 1. 加载 Solution 日志 (包含模型名)
    if os.path.exists("training_log_solution.csv"):
        df_sol = pd.read_csv("training_log_solution.csv")
        all_dfs.append(df_sol)

    # 2. 加载 Advanced 日志 (RoBERTa)，并进行折均值处理
    if os.path.exists("training_log_advanced.csv"):
        df_adv = pd.read_csv("training_log_advanced.csv")
        # 对 K-Fold 进行平均
        df_adv_avg = df_adv.groupby("epoch").mean().reset_index()
        df_adv_avg["model"] = "RoBERTa(Advanced)"
        # 确保列名一致
        all_dfs.append(df_adv_avg[["epoch", "train_loss", "val_f1", "model"]])

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


def plot_final_comparison():
    df = load_all_logs()
    if df is None:
        print("Error: No log files found. Please run training scripts first.")
        return

    # --- 1. 柱状图：最佳收益对比 ---
    best_scores = df.groupby("model")["val_f1"].max().reset_index()
    # 手动加入 Linear Baseline (因为它不在 CSV 里)
    baseline_row = pd.DataFrame({"model": ["Linear(Baseline)"], "val_f1": [0.7568]})
    summary_df = pd.concat([best_scores, baseline_row], ignore_index=True).sort_values(
        by="val_f1"
    )

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("coolwarm", len(summary_df))
    bars = plt.bar(summary_df["model"], summary_df["val_f1"], color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylim(0.65, 0.85)
    plt.title("Overall Performance Summary (Best F1 Score)", fontsize=16, pad=20)
    plt.ylabel("F1 Score")
    plt.savefig("overall_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- 2. 折线图：收敛曲线对比 ---
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df,
        x="epoch",
        y="val_f1",
        hue="model",
        marker="o",
        markersize=8,
        linewidth=2.5,
    )

    plt.axhline(y=0.7568, color="gray", linestyle="--", label="Linear Baseline")

    plt.title("Model Convergence Comparison (Val F1 per Epoch)", fontsize=16, pad=20)
    plt.ylabel("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig("convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(
        "Success: Generated 'overall_performance_comparison.png' and 'convergence_comparison.png'"
    )


if __name__ == "__main__":
    plot_final_comparison()
