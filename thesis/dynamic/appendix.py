import datetime as dt

def fs(feature_set):
    if feature_set == "hv":
        return "\\featureset{HV}"
    elif feature_set == "half-king-piece":
        return "\\featureset{King-Piece}"
    else:
        return feature_set


def make_runs_table(df):
    df = df.sort_values(by=["feature_set", "batch_size", "l1_size", "l2_size"], ascending=[False, True, True, True])
    last_fs = None

    # $\\gamma$

    table = """
\\begin{tabular}{@{} cccccccc @{}} \\toprule
\\multirow{2}{*}{\\bf Feature set} & \\multicolumn{3}{c}{\\bf Train hyperparams} & \\multicolumn{2}{c@{}}{\\bf Network} & \\multirow{2}{*}{\\makecell{\\bf Val loss\\\\\\textit{min}}} & \\multirow{2}{*}{\\makecell{\\bf Runtime\\\\\\textit{hh:mm:ss}}} \\\\
\\cmidrule(lr){2-4} \\cmidrule(l){5-6}
& \\bf Batch & \\bf LR & \\bf Gamma & \\bf L1 & \\bf L2 & \\\\
\\midrule
    """
    # table += "Feature set & Batch size & L1 size & Train loss (min) & Runtime \\\\ \\midrule\n"

    for index, row in df.iterrows():
        if last_fs is not None and last_fs != row['feature_set']:
            table += "\\midrule\n"
        last_fs = row['feature_set']

        min_loss = df[df["feature_set"] == row["feature_set"]]['Train/val_loss.min'].min()
        loss = F"{row['Train/val_loss.min']:.5f}"
        if row['Train/val_loss.min'] == min_loss:
            loss = f"\\textbf{{{loss}}}"

        table += f"{fs(row['feature_set'])} & {row['batch_size']} & {row['learning_rate']:.0e} & {row['gamma']} & {row['l1_size']} & {row['l2_size']} & {loss} & {dt.timedelta(seconds=row['Runtime'])} \\\\\n"

    table += "\\bottomrule \\end{tabular}\n"

    return table
