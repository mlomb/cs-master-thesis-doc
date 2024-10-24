import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = ['serif']

def plot_rating_over_time(data: str, output: str):
    df = pd.read_csv(data)
    df.sort_values(by='epoch', inplace=True)
    
    plt.figure(figsize=(6, 4))
    plt.errorbar(df["epoch"], df["rating"], yerr=df["error"], fmt='-o', capsize=5, label='Rating with Error')
    plt.xlim(None, 256 + 10)
    plt.xlabel('Epoch')
    plt.ylabel('Rating (relative to \\textsc{All})')
    plt.title('Rating of a network trained with the PQR method')

    plt.tight_layout()
    plt.savefig(output, format='pdf')

plot_rating_over_time(
    data='../../assets/results/pqr/pqr1_ratings.csv',
    output='./output/pqr1_ratings.pdf'
)