"""Plots for neural_network.ipynb experiment results."""
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history: dict[str, list[float]], name: str = "experiment") -> None:
    """One figure per experiment: train/val loss + val acc."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc, ax_f1) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(name)

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["val_acc"])
    ax_acc.set_title("Val Accuracy (%)")
    ax_acc.set_xlabel("epoch")
    
    ax_f1.plot(epochs, history["val_F1"])
    ax_f1.set_title("Val F1")
    ax_f1.set_xlabel("epoch")

    plt.tight_layout()
    plt.show()


def plot_comparison(results: dict[str, dict[str, list[float]]]) -> None:
    """Bar chart of best val acc + min val loss across experiments."""
    names = list(results.keys())
    best_acc = [max(h["val_acc"]) for h in results.values()]
    min_loss = [min(h["val_loss"]) for h in results.values()]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    ax_acc.bar(names, best_acc)
    ax_acc.set_title("Best Val Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.tick_params(axis="x", rotation=20)

    ax_loss.bar(names, min_loss)
    ax_loss.set_title("Min Val Loss")
    ax_loss.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def results_table(results: dict[str, dict[str, list[float]]]) -> None:
    """Summary table per experiment: max / min / final / avg / std of each metric."""
    import polars as pl

    rows: list[dict] = []
    for name, hist in results.items():
        for metric in ("train_loss", "val_loss", "val_acc"):
            vals: list[float] = hist[metric]
            rows.append({
                "experiment": name,
                "metric": metric,
                "max": round(max(vals), 4),
                "min": round(min(vals), 4),
                "final": round(vals[-1], 4),
                "avg": round(sum(vals) / len(vals), 4),
                "std": round(float(np.std(vals)), 4),
            })
    df = pl.DataFrame(rows)
    pl.Config.set_ascii_tables()  # ponytail: cp1252 console can't render polars box chars
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120):
        print(df)
    return df


if __name__ == "__main__":
    # smallest check that fails if the shapes break
    demo = {"train_loss": [1.1, 0.9], "val_loss": [1.2, 1.0], "val_acc": [40.0, 50.0]}
    plot_history(demo)
    plot_comparison({"a": demo, "b": demo})