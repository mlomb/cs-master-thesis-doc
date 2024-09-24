import datetime as dt

def fs(feature_set):
    if feature_set == "hv":
        return "\\featureset{HV}"
    elif feature_set == "half-king-piece":
        return "\\featureset{King-Piece}"
    else:
        return "\\featureset{" + feature_set + "}"


def make_runs_table(df, sort_by_elo=False):
    """Don't look at this function :)"""

    df = df.sort_values(by=["feature_set", "batch_size", "l1_size", "l2_size"], ascending=[False, True, True, True])

    if sort_by_elo:
        df = df.sort_values(by=["Perf/rating"], ascending=False)

    last_fs = None

    has_rating = "Perf/rating" in df.columns
    has_puzzles = "Puzzle/accuracy" in df.columns

    # $\\gamma$
    # \\multirow{2}{*}{\\makecell{\\bf Feature\\\\set}} &

    table = """
\\begin{tabular}{@{} cccccccc""" + ('c' if has_rating else '') + ('c' if has_puzzles else '')  + """ @{}} \\toprule
\\multirow{2}{*}{\\bf Feature set} &
\\multicolumn{3}{c}{\\bf Train hyperparams} &
\\multicolumn{2}{c@{}}{\\bf Network} &
\\multirow{2}{*}{\\makecell{\\bf Val. loss\\\\\\textit{min}}} &""" + (
"""\\multirow{2}{*}{\\makecell{\\bf Rating\\\\\\textit{elo (avg=0)}}} &""" if has_rating else "") + (
"""\\multirow{2}{*}{\\makecell{\\bf Puzzles\\\\\\textit{move acc.}}} &""" if has_puzzles else "") + """
\\multirow{2}{*}{\\makecell{\\bf Runtime\\\\\\textit{hh:mm:ss}}} \\\\
\\cmidrule(lr){2-4} \\cmidrule(l){5-6}
& \\bf Batch & \\bf LR & \\bf Gamma & \\bf L1 & \\bf L2 & \\\\
\\midrule
    """
    # table += "Feature set & Batch size & L1 size & Train loss (min) & Runtime \\\\ \\midrule\n"

    for index, row in df.iterrows():
        if last_fs is not None and last_fs != row['feature_set']:
            table += "\\midrule\n"
        last_fs = row['feature_set']

        min_loss = df['Train/val_loss.min'].min()
        loss = F"{row['Train/val_loss.min']:.5f}"
        # print(int(row['Train/val_loss.min'] * 1_00000) ,"<=", int(min_loss * 1_00000))
        if int(row['Train/val_loss.min'] * 1_00000) <= int(min_loss * 1_00000):
            loss = f"\\textbf{{{loss}}}"

        table += f"{fs(row['feature_set'])} & {row['batch_size']} & {row['learning_rate']:.0e} & {row['gamma']} & {row['l1_size']} & {row['l2_size']} & {loss}"
        
        if has_rating:
            max_rating = df['Perf/rating'].max()
            if row['Perf/rating'] >= max_rating:
                table += f" & \\textbf{{{row['Perf/rating']:.1f} $\\pm$ {row['Perf/rating_error']:.1f}}}"
            else:
                table += f" & {row['Perf/rating']:.1f} $\\pm$ {row['Perf/rating_error']:.1f}"

        if has_puzzles:
            max_puzzles = df['Puzzle/accuracy'].max()
            if row['Puzzle/accuracy'] == max_puzzles:
                table += f" & \\textbf{{{row['Puzzle/accuracy']:.4f}}}"
            else:
                table += f" & {row['Puzzle/accuracy']:.4f}"

        runtime = row['Runtime']
        runtime = (int(runtime) / row['epochs']) * 256 # rescale to 256 epochs

        table += f" & {dt.timedelta(seconds=int(runtime))} \\\\\n"

    table += "\\bottomrule \\end{tabular}\n"

    return table
