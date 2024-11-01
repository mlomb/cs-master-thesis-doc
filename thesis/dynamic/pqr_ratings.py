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

#plot_rating_over_time(
#    data='../../assets/results/pqr/pqr1_ratings.csv',
#    output='./output/pqr1_ratings.pdf'
#)

def plot_rating_over_time_series(data: str, output: str):
    # CSV columns: network, epoch, rating, error
    df = pd.read_csv(data)
    df.sort_values(by=['network','epoch'], inplace=True)

    plt.figure(figsize=(6, 4))
    for network, group in df.groupby('network'):
        if network == 'all':
            label = 'Target scores'
        elif network == 'm2':
            label = 'PQR $M=2$'
        elif network == 'm8':
            label = 'PQR $M=8$'
        elif network == 'm15':
            label = 'PQR $M=15$'

        if network == 'all' or network == 'm15':
            last_epoch_data = group.iloc[-1]
            plt.annotate('{} $\pm$ {}'.format(last_epoch_data["rating"], last_epoch_data["error"]), xy=(last_epoch_data["epoch"],last_epoch_data["rating"]), xytext=(-5, 5), ha='right', textcoords='offset points')

        plt.errorbar(group["epoch"], group["rating"], yerr=group["error"], fmt='-o', markersize=5, capsize=5, label=label)
    plt.xlim(None, 256 + 10)
    plt.xlabel('Epoch')
    plt.ylabel('Rating (relative to average)')
    plt.title('Rating of networks trained with the PQR method')

    plt.legend(title='Network')

    plt.tight_layout()
    plt.savefig(output, format='pdf')

plot_rating_over_time_series(
    data='../../assets/results/pqr/pqr_comparison.csv',
    output='./output/pqr_comparison.pdf'
)
