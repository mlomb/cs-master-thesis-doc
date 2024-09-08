# Whoever you are, whatever you're doing, stop reading this file.
# This is a mess. This file transforms the results of the runs into plots and tables for... LaTeX 💀

import pandas as pd
import humanize
import datetime as dt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True

LOSS_CMAP = "flare"
ACC_CMAP = "YlGn"

def fs(feature_set):
    if feature_set == "half-piece":
        return "\\featureset{Piece}"
    elif feature_set == "half-king-piece":
        return "\\featureset{King-Piece}"
    else:
        return feature_set

def gen_appendix_table_runs(df):
    df = df.sort_values(by=["feature_set", "batch_size", "ft_size"], ascending=[False, True, True])
    last_fs = None

    table = """
\\begin{tabular}{@{} cccccccc @{}} \\toprule
\\multirow{2}{*}{\\bf Feature set} & \\multicolumn{2}{c}{\\bf Train hyperparams} & \\multicolumn{3}{c@{}}{\\bf Network arch} & \\multirow{2}{*}{\\makecell{\\bf Train loss\\\\\\textit{min}}} & \\multirow{2}{*}{\\makecell{\\bf Runtime\\\\\\textit{hh:mm:ss}}} \\\\
\\cmidrule(lr){2-3} \\cmidrule(l){4-6}
& \\bf Batch size & \\bf Learning rate & \\bf L1 (FT) & \\bf L2 & \\bf L3 \\\\
\\midrule
    """
    # table += "Feature set & Batch size & L1 size & Train loss (min) & Runtime \\\\ \\midrule\n"

    for index, row in df.iterrows():
        if last_fs is not None and last_fs != row['feature_set']:
            table += "\\midrule\n"
        last_fs = row['feature_set']

        min_loss = df[df["feature_set"] == row["feature_set"]]['Train/loss.min'].min()
        loss = F"{row['Train/loss.min']:.5f}"
        if row['Train/loss.min'] == min_loss:
            loss = f"\\textbf{{{loss}}}"

        table += f"{fs(row['feature_set'])} & {row['batch_size']} & {row['learning_rate']} & {row['ft_size']} & {row['l1_size']} & {row['l2_size']} & {loss} & {dt.timedelta(seconds=row['Runtime'])} \\\\\n"

    table += "\\bottomrule \\end{tabular}\n"
    return table

def gen_puzzles_heatmap(df):
    #value_col = "Puzzles/moveAccuracy (Max)"
    #sns.heatmap(
    #    sub_df.pivot(index="batch_size", columns="ft_size", values=value_col),
    #    annot=True,
    #    fmt=".4f",
    #    cmap="flare",
    #    cbar=False,
    #    cbar_kws={'label': "pepito"},
    #    ax=ax,
    #    vmin=df[value_col].min(),
    #    vmax=df[value_col].max()
    #)

    # sort by loss
    df = df.sort_values(by=["Train/loss.min"], ascending=True)

    puzzles = [x for x in list(df.columns) if ("Puzzles/" in x and not "max" in x.lower() and x != "Puzzles/accuracy") or "loss.min" in x.lower()]
    df = df.transpose()

    count = 0    
    for index, row in df.iterrows():
        if index not in puzzles:
            continue
        data = np.array([list(row.values)])
        if np.isnan(data).all():
            continue
        count += 1

    f, axs = plt.subplots(count, 1, gridspec_kw={'hspace': 0}, figsize=(10, 20))

    counter = 0
    for index, row in df.iterrows():
        if index not in puzzles:
            continue
        ax = axs[counter]
        isloss = "loss" in index.lower()
        data = np.array([list(row.values)])
        if np.isnan(data).all():
            continue
        sns.heatmap(
            data=data,
            xticklabels=list(df.loc["feature_set"]) if counter == 0 else False,
            yticklabels=[index.split("/")[1]],
            annot=True,
            fmt=".4f",
            ax=ax,
            cmap=LOSS_CMAP if isloss else ACC_CMAP,
            cbar=False,
            vmin=None if isloss else 0,
            vmax=None if isloss else 1
        )
        counter += 1
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='y', labelrotation=0)
        ax.tick_params(axis='x', labelrotation=45)

    #ax.set_title(value_label)
    ##ax.xaxis.tick_top()
    ##ax.xaxis.set_label_position('top')
    #if right_ticks:
    #    ax.yaxis.tick_right()
    #    ax.yaxis.set_label_position('right')
    #
    ##ax.xaxis.set_ticks_position('none')
    ##ax.yaxis.set_ticks_position('none')
    #ax.set_xlabel("L1 size (feature transformer)")
    #ax.set_ylabel("Batch size")
    ##ax.set_title(fs(sub_df["feature-set"].iloc[0]))
    #ax.tick_params(axis='y', labelrotation=0)


def gen_baseline_tables():
    df = pd.read_csv('../../assets/results/initial_sweep.csv')

    with open('./baseline_appendix.tex', 'w') as f:
        f.write(gen_appendix_table_runs(df))
    
    def gen_heatmap(ax, sub_df, value_col, value_label, cmap, right_ticks=False):
        sns.heatmap(
            sub_df.pivot(index="batch_size", columns="ft_size", values=value_col),
            annot=True,
            fmt=".4f",
            cmap=cmap,
            cbar=False,
            cbar_kws={'label': value_label},
            ax=ax,
            vmin=df[value_col].min(),
            vmax=df[value_col].max()
        )
        ax.set_title(value_label)
        #ax.xaxis.tick_top()
        #ax.xaxis.set_label_position('top')
        if right_ticks:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        
        #ax.xaxis.set_ticks_position('none')
        #ax.yaxis.set_ticks_position('none')
        ax.set_xlabel("L1 size (feature transformer)")
        ax.set_ylabel("Batch size")
        #ax.set_title(fs(sub_df["feature-set"].iloc[0]))
        ax.tick_params(axis='y', labelrotation=0)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    gen_heatmap(ax1, df[df["feature-set"] == "half-piece"], value_col="Train/loss.min", value_label="Train loss $(min)$", cmap=LOSS_CMAP)
    gen_heatmap(ax2, df[df["feature-set"] == "half-king-piece"], value_col="Train/loss.min", value_label="Train loss $(min)$", cmap=LOSS_CMAP, right_ticks=True)

    gen_heatmap(ax3, df[df["feature-set"] == "half-piece"], value_col="Puzzles/moveAccuracy (Max)", value_label="Puzzle move accuracy $(max)$", cmap=ACC_CMAP)
    gen_heatmap(ax4, df[df["feature-set"] == "half-king-piece"], value_col="Puzzles/moveAccuracy (Max)", value_label="Puzzle move accuracy $(max)$", cmap=ACC_CMAP, right_ticks=True)

    add_headers(
        fig,
        col_headers=["\\textsc{Piece}", "\\textsc{King-Piece}"],
        fontweight="bold",
        fontsize="16",
        col_pad=30
    )

    line = plt.Line2D((.5,.5),(0.03, 0.98), color="k", linewidth=1)
    fig.add_artist(line)

    plt.tight_layout()
    plt.savefig("./baselines_comparison.pdf", format='pdf')
    plt.close()


def gen_heatmaps():
    df = pd.read_csv('../../assets/results/axes_sweep.csv')
    gen_puzzles_heatmap(df[(df["batch_size"] == 4096) & (df["ft_size"] == 2048)])
    plt.tight_layout()
    plt.savefig("./axes_puzzles.pdf", format='pdf')
    plt.close()


def gen_quantization_error():
    df = pd.read_csv('../../assets/misc/quant_errors.csv')
    df["err"] = (df["expected_output"] - df["output"])
    df["err_abs"] = (df["expected_output"] - df["output"]).abs()

    sns.displot(df, x='err', height=3, aspect=1.3, bins=35) # , linewidth=0
    plt.xlabel('Absolute error of quantized model')
    plt.ylabel('Count')
    plt.xlim(-100, 100)
    #plt.ylim(0, 3000)
    plt.savefig("./quant_errors.pdf", format='pdf')
    plt.close()

# https://stackoverflow.com/a/71887460/2840384
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


# gen_baseline_tables()
# gen_heatmaps()
gen_quantization_error()
