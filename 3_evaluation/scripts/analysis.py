import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scripts.plots import PALETTE


def plot_breakeven(df, save_fn=None):
    X = df[["quant_bits", "thinking_budget"]].values
    model = LinearRegression().fit(X, df["accuracy"].values)
    a_q, a_tau = model.coef_
    intercept  = model.intercept_

    accuracy_gap        = a_q * 4
    tokens_to_breakeven = accuracy_gap / a_tau

    print(f"Accuracy lost (8-bit → 4-bit)            : {accuracy_gap:.4f}")
    print(f"Extra tokens needed to recover that gap  : {tokens_to_breakeven:.0f}")
    print(f"Tested budget range                      : 64 - 512 tokens")
    print(f"Max tested budget covers                 : {512 / tokens_to_breakeven * 100:.0f}% of the required gap\n")

    print("Breakeven budgets (4-bit tokens needed to match 8-bit at a given budget):")
    for budget_8bit in [64, 128, 256, 512]:
        acc_8bit         = a_q * 8 + a_tau * budget_8bit + intercept
        budget_4bit_even = (acc_8bit - a_q * 4 - intercept) / a_tau
        reachable        = "within tested range" if budget_4bit_even <= 512 else f"needs {budget_4bit_even:.0f} tokens (above 512)"
        print(f"  8-bit @ {budget_8bit:>3} tokens → 4-bit needs {budget_4bit_even:>5.0f} tokens  [{reachable}]")

    budget_range  = np.linspace(0, 700, 300)
    acc_8bit_line = a_q * 8 + a_tau * budget_range + intercept
    acc_4bit_line = a_q * 4 + a_tau * budget_range + intercept

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(budget_range, acc_8bit_line, color=PALETTE[0], label="8-bit (predicted)")
    ax.plot(budget_range, acc_4bit_line, color=PALETTE[3], label="4-bit (predicted)", linestyle="--")
    ax.axvline(512, color="gray", linewidth=0.8, linestyle=":", label="max tested budget (512)")
    ax.axvline(64 + tokens_to_breakeven, color=PALETTE[1], linewidth=0.8, linestyle="-.",
               label=f"4-bit breakeven vs 8-bit@64 ({64 + tokens_to_breakeven:.0f} tokens)")
    ax.scatter(df[df.quant_bits == 8].thinking_budget, df[df.quant_bits == 8].accuracy,
               color=PALETTE[0], s=8, alpha=0.3, zorder=3)
    ax.scatter(df[df.quant_bits == 4].thinking_budget, df[df.quant_bits == 4].accuracy,
               color=PALETTE[3], s=8, alpha=0.3, zorder=3)
    ax.set(xlabel="Thinking budget (tokens)", ylabel="Accuracy",
           title="Predicted accuracy: 4-bit vs 8-bit")
    ax.legend(fontsize=7, frameon=False)
    if save_fn:
        save_fn("breakeven_accuracy")
    plt.tight_layout()