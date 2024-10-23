import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = ['serif']

def plot_pqr(file: str, output: str, title: str):
    df = pd.read_csv(file) # names=['p','q','r']
    df = df.sample(10000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title, fontsize=18)

    # A
    ax1 = sns.scatterplot(data=df, x='p', y='q', alpha=0.5, ax=ax1)
    ax1.plot([-2000,2000], [2000, -2000], 'k-', alpha=0.75, color='red')
    ax1.set_xlim(-2000, 2000)
    ax1.set_ylim(-2000, 2000)
    ax1.set_box_aspect(1)
    ax1.set_xlabel('$f(p)$')
    ax1.set_ylabel('$f(q)$')
    ax1.title.set_text('$f(q)$ vs $f(p)$')

    df['diff'] = df['r'] - df['q']
    sns.scatterplot(data=df, x='q', y='diff', alpha=0.1, ax=ax2)
    sns.kdeplot(data=df, x='q', y='diff', ax=ax2)
    ax2.axhline(0, color='red')
    ax2.set_ylim(-2000,2000)
    ax2.set_xlim(-2000,2000)
    ax2.set_box_aspect(1)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel('$f(q)$')
    ax2.set_ylabel('$f(r) - f(q)$')
    ax2.title.set_text('Difference between $f(r)$ and $f(q)$')

    plt.tight_layout()
    plt.savefig(output, format='pdf')


plot_pqr(
    file='../../assets/results/pqr/pqr_eval.csv',
    output='./output/pqr_eval.pdf',
    title='PQR analysis for a network trained with target scores'
)
