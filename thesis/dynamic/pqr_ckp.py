import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = ['serif']

def plot_rating_over_time_series(data: str, output: str):
    # CSV columns: lr, epoch, rating, error
    df = pd.read_csv(data)
    df.sort_values(by=['lr','epoch'], ascending=[True, True], inplace=True)
    G = df.groupby('lr')

    # sort G by lr
    G = sorted(G, key=lambda x: x[0])

    plt.figure(figsize=(6, 4))
    for lr, group in G:
        if lr == 'all':
            label = 'Target scores'
            color = 'm'
        elif lr == 'P00':
            label = 'PQR $p=0.00$'
            color = 'orange'
        elif lr == 'P25':
            label = 'PQR $p=0.25$'
            color = 'blue'
        elif lr == 'P50':
            label = 'PQR $p=0.50$'
            color = 'green'
        elif lr == 'P75':
            label = 'PQR $p=0.75$'
            color = 'red'
        else:
            label = lr
            color = None

        if lr == 'all' or lr == 'P00' or lr == 'P75':
            last_epoch_data = group.iloc[-1]
            plt.annotate('\\bf{} $\pm$ {}'.format(last_epoch_data["rating"], last_epoch_data["error"]), xy=(last_epoch_data["epoch"],last_epoch_data["rating"]), xytext=(-5, -15 if lr == 'P75' else 5), ha='right', textcoords='offset points', color=color)

        X = group["epoch"]
        Y = group["rating"]
        E = group["error"]

        # add point at 0
        if False:
            X = np.concatenate(([0], X))
            Y = np.concatenate(([0], Y))
            E = np.concatenate(([0], E))

        plt.errorbar(X, Y, yerr=E, fmt='-o', markersize=4, capsize=4, label=label, color=color)

    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('Epoch')
    plt.ylabel('Rating (relative to start checkpoint)')
    plt.title('Rating of networks fine-tuned with the PQR method')
    #plt.xlim(0, None)

    plt.legend(title='Learning rate', loc='lower right')

    # horizontal line at 0
    plt.axhline(0, color='k', linestyle='--')


    plt.tight_layout()
    plt.savefig(output, format='pdf')

plot_rating_over_time_series(
    data='../../assets/results/pqr/pqr_ckp.csv',
    output='./output/pqr_ckp.pdf'
)
