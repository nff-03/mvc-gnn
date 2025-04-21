import os

import matplotlib.pyplot as plt
import pandas as pd

# Load Excel file and relevant sheets
file_path = "Final Results (5-degree graphs).xlsx"
sheets = {
    "ILP": "Average ILP",
    "Heuristic": "Average Heuristic",
    "Approximate": "Average Approx.",
    "GNN": "Average GNN"
}

# Marker styles and colors for each method
styles = {
    "ILP": {"marker": "s", "color": "#1f77b4"},
    "Heuristic": {"marker": "o", "color": "#ff7f0e"},
    "Approximate": {"marker": "^", "color": "#2ca02c"},
    "GNN": {"marker": "d", "color": "#9467bd"},
}

# Read all data into a dictionary and ensure numeric columns
data = {}
for name, sheet in sheets.items():
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric if possible
    data[name] = df

# Create plots
for name, df in data.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    x_nodes = df["Num Nodes"].to_numpy()
    y_solution = df["Solution Size"].to_numpy()

    # Solution Size plot
    ax1.plot(x_nodes, y_solution,
             label=name,
             marker=styles[name]["marker"],
             color=styles[name]["color"],
             linestyle='-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Number of Nodes (n)")
    ax1.set_ylabel("Solution Size")
    ax1.set_title(f"{name}: Solution Size vs Number of Nodes (3-degree Graph)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Inset: Approximation Ratio (top-right)
    if "Approximation Ratio" in df.columns:
        inset_ax = ax1.inset_axes([0.05, 0.55, 0.35, 0.35])
        inset_ax.plot(x_nodes, df["Approximation Ratio"].to_numpy(),
                      marker=styles[name]["marker"],
                      color=styles[name]["color"],
                      linestyle='-')
        inset_ax.set_xscale('log')
        inset_ax.set_ylim(df["Approximation Ratio"].min() * 0.995,
                          df["Approximation Ratio"].max() * 1.005)
        inset_ax.set_title("Approx. Ratio", fontsize=10)
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(True, linestyle="--", linewidth=0.5)

    # Inset: MIP Gap (top-left, only for ILP)
    if name == "ILP" and "MIP Gap" in df.columns:
        mip_ax = ax1.inset_axes([0.05, 0.55, 0.35, 0.35])
        mip_ax.plot(x_nodes, df["MIP Gap"].to_numpy(),
                    marker=styles[name]["marker"],
                    color="#d62728",
                    linestyle='-')
        mip_ax.set_xscale('log')
        mip_ax.set_ylim(df["MIP Gap"].min() * 0.995,
                        df["MIP Gap"].max() * 1.005)
        mip_ax.set_title("MIP Gap", fontsize=10)
        mip_ax.tick_params(labelsize=8)
        mip_ax.grid(True, linestyle="--", linewidth=0.5)

    # Runtime plot
    if name == "GNN":
        ax2.plot(x_nodes, df["Runtime (s)"].to_numpy(), label="Total Runtime", marker='o', color="#9467bd")
        ax2.plot(x_nodes, df["GNN Training Runtime (s)"].to_numpy(), label="Training Time", marker='s', color="#2ca02c")
        ax2.plot(x_nodes, df["Post-Processing Runtime (s)"].to_numpy(), label="Post-Processing Time", marker='^', color="#ff7f0e")
    else:
        ax2.plot(x_nodes, df["Runtime (s)"].to_numpy(),
                 label=name,
                 marker=styles[name]["marker"],
                 color=styles[name]["color"],
                 linestyle='-')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Number of Nodes (n)")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title(f"{name}: Runtime vs Number of Nodes (3-degree Graph)")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_file = f"{name.lower()}_results_plot.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved {output_file}")
