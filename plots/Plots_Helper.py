import matplotlib.pyplot as plt
from plots.Custom_plot_style import apply_custom_theme
import os

def plot_ensemble_metrics(ensemble, single, mnist=True):
    """
    Plots evaluation metrics vs. ensemble size from a DataFrame and compares them to a fixed BNN baseline.

    Parameters:
    -----------
    ensemble : pd.DataFrame
        Must contain columns:
        - "Ensemble Size"
        - "Accuracy", "NLL", "Brier Score", "Predictive Entropy", "ECE"

    single : dict
        Reference metric values from a single BNN model. Keys must match column names in df.

    mnist : bool (default=True)
        If True, filenames end with "_MNIST.png", otherwise "_CIFAR10.png"
    """
    import matplotlib.ticker as mtick

    ensemble_sizes = ensemble["Ensemble Size"].tolist()
    metric_columns = [col for col in ensemble.columns if col != "Ensemble Size"]
    suffix = "_MNIST.png" if mnist else "_CIFAR10.png"

    plots = []

    for metric_name in metric_columns:
        values = ensemble[metric_name].tolist()
        single_value = single[metric_name]

        is_percent_metric = metric_name == "Accuracy"  # ✅ Nur Accuracy in %
        if is_percent_metric:
            values = [v * 100 for v in values]
            single_value *= 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ensemble_sizes, values, marker='o', color='blue', label='DE-BNN', linewidth=2.5, markersize=8)
        ax.axhline(y=single_value, color='red', linestyle='--', label='Single-BNN', linewidth=2.5)
        ax.set_xlabel("Ensemble Size")
        ylabel = f"{metric_name} (%)" if is_percent_metric else metric_name
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric_name} vs. Ensemble Size")
        ax.set_xticks(ensemble_sizes)

        # Setzt Formatierung je nach Metrik
        if is_percent_metric:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        elif metric_name == "Brier Score":
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        else:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        ax.legend(fontsize=20)
        ax.grid(True)
        fig.tight_layout()
        apply_custom_theme(ax)

        """# Save to plots/Saved_Plots
        filename = f"plots/Saved_Plots/{metric_name.replace(' ', '_').lower()}{suffix}"
        fig.savefig(filename, bbox_inches="tight", dpi=300)"""
        plots.append((fig, metric_name, suffix))
        plt.show()

    return plots

def save_plots(plots, output_dir="plots/Saved_Plots", dpi=300, file_suffix=""):
    """
    Saves figures created by `plot_ensemble_metrics`.

    Parameters
    ----------
    plots : list of (figure, metric_name, suffix)
        Tuples returned by `plot_ensemble_metrics`.
    output_dir : str
        Directory where plots should be saved.
    dpi : int
        Image resolution.
    file_suffix : str
        Extra tag to append just *before* the file extension (e.g. "run2").
        If given, a leading underscore is automatically added.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure our extra tag starts with "_" (unless empty)
    if file_suffix:
        file_suffix = "_" + file_suffix.lstrip("_")

    for fig, metric_name, suffix in plots:
        # `suffix` comes from plot_ensemble_metrics, e.g. "_MNIST.png"
        if suffix.lower().endswith(".png"):
            dataset_tag, ext = suffix[:-4], suffix[-4:]  # "_MNIST", ".png"
        else:
            dataset_tag, ext = suffix, ""

        base_name = (
            f"{metric_name.replace(' ', '_').lower()}"
            f"{dataset_tag}{file_suffix}{ext}"
        )
        path = os.path.join(output_dir, base_name)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)  # free memory
