import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True


def quantization_error():
    df = pd.read_csv('../../assets/results/quant_errors.csv')
    df["err"] = (df["expected_output"] - df["output"])
    df["rel_err"] = (df["err"] / df["expected_output"]).abs()

    sns.displot(df, x='err', height=3, aspect=1.3, bins=20) # , linewidth=0
    plt.xlabel('Error of quantized model')
    plt.ylabel('Count')

    # draw quantile 0.5 line
    plt.axvline(x=df["err"].mean(), color='r', linestyle='dotted')
    plt.text(df["err"].mean(), 11500, 'mean', color='r', ha='left', va='top', rotation=90)
    print(f"Quantization error: min: {df["output"].min()} max: {df["output"].max()} mean: {df["output"].mean()}")

    #plt.ylim(0, 3000)
    plt.savefig("./output/quant_errors.pdf", format='pdf')
    plt.close()

quantization_error()
