import pandas as pd
import humanize
import datetime as dt

def fs(feature_set):
    if feature_set == "half-piece":
        return "\\featureset{Piece}"
    elif feature_set == "half-king-piece":
        return "\\featureset{King-Piece}"
    else:
        return feature_set

def gen_baseline_tables():
    df = pd.read_csv('../../assets/results/initial_sweep.csv')

    def gen_table(df, caption):
        # extract data of columns "batch_size" and "ft_size" and the value "Train/loss.min"
        data = {}

        for index, row in df.iterrows():
            batch_size = row['batch_size']
            ft_size = row['ft_size']
            train_loss = row['Train/loss.min']
            data[(batch_size, ft_size)] = train_loss

        batch_sizes = df['batch_size'].unique()
        batch_sizes.sort()
        ft_sizes = df['ft_size'].unique()
        ft_sizes.sort()

        print(data, batch_sizes, ft_sizes)

        # build a LaTeX table
        table = "\\begin{table}[H]\n"
        table += "\\centering\n"
        table += "\\begin{tabular}{|c|"
        for ft_size in ft_sizes:
            table += "c|"
        table += "}\n"
        table += "\\hline\n"
        table += "Batch size / FT size & "
        for ft_size in ft_sizes:
            table += f"{ft_size} & "
        table = table[:-2] + "\\\\\n"
        table += "\\hline\n"
        for batch_size in batch_sizes:
            table += f"{batch_size} & "
            for ft_size in ft_sizes:
                if (batch_size, ft_size) in data:
                    value = f"{data[(batch_size, ft_size)]:.4f}"
                else:
                    value = "N/A"

                table += f"{value} & "
            table = table[:-2] + "\\\\\n"
        table += "\\hline\n"
        table += "\\end{tabular}\n"
        table += "\\caption{" + caption + "}\n"
        table += "\\label{tab:baseline}\n"
        table += "\\end{table}\n"

        return table

    def gen_appendix_table_runs(df):
        df = df.sort_values(by=["feature_set", "batch_size", "ft_size"], ascending=[False, True, True])
        last_fs = None

        table = """
\\begin{tabular}{@{} cccccccc @{}} \\toprule
\\multirow{2}{*}{Feature set} & \\multicolumn{2}{c}{Train hyperparams} & \\multicolumn{3}{c@{}}{Network arch} & \\multirow{2}{*}{\\makecell{Train loss\\\\\\textit{min}}} & \\multirow{2}{*}{\\makecell{Runtime\\\\\\textit{hh:mm:ss}}} \\\\
\\cmidrule(lr){2-3} \\cmidrule(l){4-6}
& Batch size & Learning rate & L1 (FT) & L2 & L3 \\\\
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

    with open('./baseline_tables.tex', 'w') as f:
        f.write(gen_table(df[df["feature-set"] == "half-piece"], "\\featureset{Piece}"))
        f.write(gen_table(df[df["feature-set"] == "half-king-piece"], "\\featureset{King-Piece}"))

    with open('./baseline_appendix.tex', 'w') as f:
        f.write(gen_appendix_table_runs(df))

gen_baseline_tables()
