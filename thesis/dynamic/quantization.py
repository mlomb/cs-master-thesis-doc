import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = ['serif']


def quantization_error():
    df = pd.read_csv('../../assets/results/quant_errors.csv')
    df = df.sample(n=100000)

    df["diff"] = df["expected_output"] - df["output"]

    def to_wdl(x):
        return 1/(1 + np.exp(-x / 361))

    df["diff"] = to_wdl(df["expected_output"]) - to_wdl(df["output"])

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8, 3))
    sns.kdeplot(df['output'], ax=ax1, bw_adjust=0.2, legend="Quantized model")
    sns.kdeplot(df['expected_output'], ax=ax1, bw_adjust=0.2, legend="Float model")

    ax1.set_xlabel('Evaluation')
    ax1.set_ylabel('Density (log)')
    ax1.set_yscale('log')
    #ax.set_xlim(-1000, 1000)
    #ax.set_xlim(-500, 500)
    ax1.set_ylim(1e-5, 1e-2)
    ax1.legend(labels=['Quantized model', 'Float model'])
    ax1.set_title('Evaluation distribution')

    sns.histplot(df, x='diff',bins=60, ax=ax2) # , linewidth=0
    plt.xlabel('Error in WDL (float - quantized)')
    plt.ylabel('Count')
    plt.title('Error distribution in WDL')

    # draw quantile 0.5 line
    plt.axvline(x=df["diff"].mean(), color='r', linestyle='dotted')
    plt.text(df["diff"].mean(), 18000, 'mean', color='r', ha='left', va='top', rotation=90)

    plt.tight_layout()
    plt.savefig("./output/quant_errors.pdf", format='pdf')
    plt.close()

quantization_error()
