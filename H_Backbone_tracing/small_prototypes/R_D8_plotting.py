import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_emd_line(line):
    """
    Extract EMD id and resolution from lines like:
    EMD-4649: ... EMD-4649_re_4.34
    """
    line = line.strip()

    emd_start = line.find("EMD-")
    emd_end = line.find(":", emd_start)

    if emd_start == -1 or emd_end == -1:
        return None, None

    emd_id = int(line[emd_start + 4:emd_end])

    re_tag = "_re_"
    re_start = line.find(re_tag)

    if re_start == -1:
        return emd_id, None

    resolution_str = line[re_start + len(re_tag):].split()[0]
    return emd_id, float(resolution_str)


def parse_metrics_line(line):
    """
    Extract metrics tuple from:
    metrics (...): (a, b, c, d, e, f)
    """
    line = line.strip()

    second_paren = line.find("(", line.find("(") + 1)
    end = line.find(")", second_paren)

    if second_paren == -1 or end == -1:
        return None

    values = [v.strip() for v in line[second_paren + 1:end].split(",")]

    return {
        "precision": float(values[0]),
        "recall": float(values[1]),
        "f1": float(values[2]),
        "tp": int(values[3]),
        "fp": int(values[4]),
        "fn": int(values[5]),
    }


def parse_full_log(log_path):
    results = []

    current_emd = None
    current_res = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # detect EMD line
            if line.startswith("EMD-"):
                current_emd, current_res = parse_emd_line(line)

            # detect metrics line
            elif line.startswith("metrics"):
                metrics = parse_metrics_line(line)

                if metrics is None:
                    continue

                results.append({
                    "emd_id": current_emd,
                    "resolution": current_res,
                    **metrics
                })

    return pd.DataFrame(results)

# Plotting functions
def scatter_plot(df, x_col, y_col, title=None, figsize=(8, 6), annotate=False):
    """
    Create a scatterplot from a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame
    - x_col: str, column name for x-axis
    - y_col: str, column name for y-axis
    - title: optional plot title
    - figsize: tuple, figure size
    - annotate: if True, labels points with index
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df[x_col], df[y_col], s=100, alpha=0.5)

    ax.set_xlabel(x_col.capitalize(), fontsize=24)
    ax.set_ylabel(y_col.capitalize(), fontsize=24)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlim([2, 10])
    ax.set_xticks(np.arange(2, 11, step=2))
    ax.set_yticks(np.arange(0, 1.1, step=0.1), minor=True)
    # ax.yaxis.set_tick_params(which='both', width=1.5)

    if title:
        ax.set_title(title, fontsize=28)

    # Optional: label points with index (useful for debugging/outliers)
    if annotate:
        for i, row in df.iterrows():
            ax.annotate(i, (row[x_col], row[y_col]), fontsize=8, alpha=0.7)

    # Vertical lines
    plt.axvline(x=4, linestyle='-.', alpha=0.5, ymin=0.01, ymax=0.99)
    plt.axvline(x=6, linestyle="-.", alpha=0.5, ymin=0.01, ymax=0.99)
    # Horizontal line
    plt.axhline(y=0.8, linestyle ="--", alpha=0.5, xmin=0.01, color="red")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # log_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\test_DELETE_THESE\clustering_eval_log_DUMMY.txt"
    log_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\test_DELETE_THESE\clustering_eval_log.txt"
    log_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\test_DELETE_THESE2\clustering_eval_log.txt"
    log_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\clustering_figure\clustering_eval_log.txt"
    log_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\clustering_figure\clustering_eval_log2.txt"
    
    metrics_df = parse_full_log(log_file)
    # print(metrics_df)
    print(metrics_df.sort_values("resolution"))

    # scatter_plot(metrics_df, "precision", "recall", title="Precision vs Recall")

    scatter_plot(metrics_df, "resolution", "precision", title="Resolution vs Precision")
    scatter_plot(metrics_df, "resolution", "recall", title="Resolution vs Recall")
    scatter_plot(metrics_df, "resolution", "f1", title="Resolution vs F1")

    # scatter_plot(metrics_df, "tp", "fp", title="TP vs FP")