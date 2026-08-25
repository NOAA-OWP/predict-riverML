import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import argparse

def plot_model_fit(data: pd.DataFrame, target_column: str, inference_column: str, plot_label: str):
    """
    Generates and saves a scatter plot comparing model predictions against target values,
    faceted by flowline type.

    Args:
        data (pd.DataFrame): The input DataFrame. It must contain the columns specified by
                             target_column, inference_column, and 'flowline_type'.
        target_column (str): The name of the column containing the ground truth values.
        inference_column (str): The name of the column containing the predicted values.
        plot_label (str): A descriptive name for the output plot file (without extension).
    """
    y_label_map = {
        'owp_dingman_r': "Predicted r Shape Values (log scale)",
        'owp_tw_bf': "Predicted Width Values (log scale)",
        'owp_y_bf': "Predicted Depth Values (log scale)"
    }
    
    x_label = f"{target_column.replace('_', ' ').title()} (log scale)"
    y_label = y_label_map.get(inference_column, f"{inference_column.replace('_', ' ').title()} (log scale)")
    
    sns.set_theme(style="whitegrid", palette="rainbow")

    g = sns.relplot(
        data=data,
        x=target_column,
        y=inference_column,
        col="flowline_type",
        hue="flowline_type",
        kind="scatter",
        s=70,
        alpha=0.6,
        col_wrap=3,
        height=5,
        aspect=1,
        legend=False,
    )

    for ax, flow_type in zip(g.axes.flat, g.col_names):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=2, label="Perfect Fit (1:1)")

        sub_df = data[data['flowline_type'] == flow_type]
        
        if not sub_df.empty and len(sub_df) > 1:
            r2 = r2_score(sub_df[target_column], sub_df[inference_column])
            ax.text(0.05, 0.95, f'$R^2 = {r2:.2f}$',
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

        ax.legend(loc='lower right')

    g.fig.suptitle(f"Model Goodness of Fit for '{inference_column}' by Flowline Type", y=1.02, fontsize=17, fontweight='bold')
    g.set_axis_labels(x_label, y_label, fontsize=14)
    g.set_titles("Flowline: {col_name}", size=16)
    
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    file_path = os.path.join(output_dir, f"{plot_label}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(file_path, dpi=300)
    plt.close(g.fig)
    print(f"Plot saved successfully to: {file_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate and save a model goodness-of-fit plot from a Parquet file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Required. Path to the input Parquet file."
    )
    
    parser.add_argument(
        "--target_column",
        type=str,
        default='target',
        help="The column name for the target (ground truth) values.\n(default: 'target')"
    )
    
    parser.add_argument(
        "--inference_column",
        type=str,
        default='owp_dingman_r',
        choices=['owp_dingman_r', 'owp_tw_bf', 'owp_y_bf'],
        help="The prediction column to plot on the y-axis.\n"
             "Choices are: 'owp_dingman_r', 'owp_tw_bf', 'owp_y_bf'.\n(default: 'owp_dingman_r')"
    )
    
    parser.add_argument(
        "--plot_label",
        type=str,
        default='model_fit_plot',
        help="The filename for the output plot (without the .png extension).\n(default: 'model_fit_plot')"
    )

    args = parser.parse_args()

    try:
        print(f"Reading data from: {args.data_path}")
        input_df = pd.read_parquet(args.data_path)
        
        # Basic validation
        required_cols = [args.target_column, args.inference_column, 'flowline_type']
        if not all(col in input_df.columns for col in required_cols):
            missing = set(required_cols) - set(input_df.columns)
            raise ValueError(f"Input data is missing one or more required columns: {missing}")

        print(f"Generating plot for target='{args.target_column}' and inference='{args.inference_column}'")
        plot_model_fit(
            data=input_df,
            target_column=args.target_column,
            inference_column=args.inference_column,
            plot_label=args.plot_label
        )

    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {args.data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
