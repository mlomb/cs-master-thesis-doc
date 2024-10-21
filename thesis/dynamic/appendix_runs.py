import pandas as pd
import datetime as dt

def fs_part(fs: str):
    fs = fs.upper()
    if fs == "HV":
        fs = "All"
    if fs == "ALL":
        fs = "All"

    # :)
    if fs == "PH":
        fs = "PV"
    elif fs == "PV":
        fs = "PH"

    return "\\featureset{" + fs + "}"

def fs(feature_set: str):
    # split by +
    parts = feature_set.split("+")
    return " + ".join([fs_part(part) for part in parts])



def make_runs_table(sweep_path, rating_path, output_path, caption):
    # "Name","Train/train_loss.min"
    # "4-eval_16384_(hv+h+v+d1+d2[1320]→512)x2→32→1","0.00300217384526196"
    # "3-eval_16384_(hv+h+v+d1+d2[1320]→512)x2→32→1","0.0030013718466832258"
    df_sweep = pd.read_csv(sweep_path)

    # "Name","Perf/rating","Perf/rating_error"
    # "256-4-eval_16384_(h+v[192]→512)x2→32→1",7.4,5.2
    # "192-3-eval_16384_(h+v[192]→512)x2→32→1",4.8,4.8
    df_ratings = pd.read_csv(rating_path)
    
    # add two columns to df_sweep: Perf/rating192 and Perf/rating256
    for index, row in df_sweep.iterrows():
        name = row['Name']
        for epoch in [192, 256]:
            rating = df_ratings[df_ratings['Name'] == f"{epoch}-{name}"]
            df_sweep.at[index, f'Perf/rating{epoch}'] = rating['Perf/rating'].values[0]
            df_sweep.at[index, f'Perf/rating_error{epoch}'] = rating['Perf/rating_error'].values[0]
    
    dfgb = df_sweep.groupby('feature_set')    
    dfs = [dfgb.get_group(x) for x in dfgb.groups]


    note = "\\textbf{Batch size}: 16384, \\textbf{LR}: 5e-04, \\textbf{Gamma}: 0.99, \\textbf{L1}: 512, \\textbf{L2}: 32"

    table = """
    \\begin{table}[H]
\\caption{""" + caption + """}
\\centering
\\begin{adjustbox}{center}
\\begin{tabular}{@{} cccc|cc @{}}
\\toprule
\\bf \\multirow{2}{*}{Feature set} & \\bf \\multirow{2}{*}{Run} & \\bf Val. loss & \\bf Runtime & \\bf Rating @ 192 & \\bf Rating @ 256 \\\\
 &  & \\textit{min} & \\textit{hh:mm:ss} & \\textit{TC=100ms/m} & \\textit{TC=100ms/m} \\\\
\\midrule
    """

    for k, df in enumerate(dfs):
        if k > 0:
            table += "\\midrule\n"

        df = df.sort_values(by=["feature_set", "run"], ascending=[False, True])

        min_loss = df['Train/val_loss.min'].min()
        max_rating = max(df['Perf/rating192'].max(), df['Perf/rating256'].max())
        
        for index, row in df.iterrows():
            if row['run'] == 1:
                table += "\\multirow{" + str(len(df)) + "}{*}{"
                table += fs(row['feature_set'])
                table += "}"

            table += f" & "
            table += str(row['run'])

            loss = float(row['Train/val_loss.min'])
            is_best_loss = int(row['Train/val_loss.min'] * 1_000_000) <= int(min_loss * 1_000_000)
            table += f" & "
            table += f"\\bf" if is_best_loss else ""
            table += f"{loss:.6f}"

            runtime = row['Runtime']
            runtime = dt.timedelta(seconds=int(runtime))
            table += f" & "
            table += str(runtime)

            for epoch in [192, 256]:
                rating = row[f'Perf/rating{epoch}']
                rating_error = row[f'Perf/rating_error{epoch}']

                table += f" & "
                table += f"\\bf" if rating >= max_rating else ""
                table += f"{rating:.1f} $\\pm$ {rating_error:.1f}"

            table += "\\\\\n"

    table += "\\toprule\n"
    table += "\\multicolumn{6}{c}{\\makecell{" + note + "}} \\\\\n"
    table += "\\end{tabular}\n"
    table += "\\end{adjustbox}\n"
    table += "\\end{table}\n"

    with open(output_path, 'w') as f:
        f.write(table)


def make_final_table(sweep_path, rating_path, output_path, caption):
    df_sweep = pd.read_csv(sweep_path)
    df_ratings = pd.read_csv(rating_path)
    df = pd.merge(df_sweep, df_ratings, on="Name", how='right')

    df = df.sort_values(by=["Perf/rating"], ascending=[False])

    note = "\\textbf{Batch size}: 16384, \\textbf{LR}: 5e-04, \\textbf{Gamma}: 0.99, \\textbf{L1}: 512, \\textbf{L2}: 32"

    table = """
    \\begin{table}[H]
\\caption{""" + caption + """}
\\centering
\\begin{adjustbox}{center}
\\begin{tabular}{@{} cccccc @{}}
\\toprule
\\bf \\multirow{2}{*}{Feature set} & \\bf \\multirow{2}{*}{Run} & \\bf \\multirow{2}{*}{Epoch} & \\bf Val. loss  & \\bf Rating & \\bf TBD \\\\
 &  &  & \\textit{min}  & \\textit{TC=100ms/m} &  \\\\
\\midrule
    """

    min_loss = df['Train/val_loss.min'].min()
    max_rating = df['Perf/rating'].max()
    
    for index, row in df.iterrows():
        table += fs(row['feature_set'])

        table += f" & "
        table += str(row['run'])

        table += f" & "
        table += str(row['Perf/epoch'])

        loss = float(row['Train/val_loss.min'])
        is_best_loss = int(row['Train/val_loss.min'] * 1_000_000) <= int(min_loss * 1_000_000)
        table += f" & "
        table += f"\\bf" if is_best_loss else ""
        table += f"{loss:.6f}"

        rating = row[f'Perf/rating']
        rating_error = row[f'Perf/rating_error']
        table += f" & "
        table += f"\\bf" if rating >= max_rating else ""
        table += f"{rating:.1f} $\\pm$ {rating_error:.1f}"

        table += "\\\\\n"

    table += "\\toprule\n"
    table += "\\multicolumn{6}{c}{\\makecell{" + note + "}} \\\\\n"
    table += "\\end{tabular}\n"
    table += "\\end{adjustbox}\n"
    table += "\\end{table}\n"


    with open(output_path, 'w') as f:
        f.write(table)

# Experiment 2 - Axes
make_runs_table(
    '../../assets/results/exp2_axes/sweep.csv',
    '../../assets/results/exp2_axes/rating_runs.csv',
    './output/exp2_axes_runs.tex',
    caption="Axis feature sets preliminar runs"
)
make_final_table(
    '../../assets/results/exp2_axes/sweep.csv',
    '../../assets/results/exp2_axes/rating_final.csv',
    './output/exp2_axes_final.tex',
    caption="Axis feature sets final results"
)

make_runs_table(
    '../../assets/results/exp3_pairwise/sweep.csv',
    '../../assets/results/exp3_pairwise/rating_runs.csv',
    './output/exp3_pairwise_runs.tex',
    caption="Pairwise feature sets preliminar runs"
)
make_final_table(
    '../../assets/results/exp3_pairwise/sweep.csv',
    '../../assets/results/exp3_pairwise/rating_final.csv',
    './output/exp3_pairwise_final.tex',
    caption="Pairwise feature sets final results"
)

# \\multirow{2}{*}{\\bf Feature set} &
# \\multirow{2}{*}{\\makecell{\\bf Val. loss\\\\\\textit{min}}} &""" + (
# """\\multirow{2}{*}{\\makecell{\\bf Rating\\\\\\textit{elo (""" + "asd" + """)}}} &""" if has_rating else "") + (
# """\\multirow{2}{*}{\\makecell{\\bf Puzzles\\\\\\textit{move acc.}}} &""" if has_puzzles else "") + """
# \\multirow{2}{*}{\\makecell{\\bf Runtime\\\\\\textit{hh:mm:ss}}} \\\\
# \\cmidrule(lr){2-4} \\cmidrule(l){5-6}
