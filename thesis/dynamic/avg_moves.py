import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = ['serif']

df = pd.read_csv("../../assets/results/pqr/avg_moves.csv")
# CSV columns: color, fullmove, legal_moves, count
df = df[df["color"] == "White"]
df["fullmove"] = pd.to_numeric(df["fullmove"])
df["legal_moves"] = pd.to_numeric(df["legal_moves"])

plt.figure(figsize=(6, 4))

Xr = np.arange(0, 101)
Yr = np.arange(0, 60)

# plot distribution of legal moves
X, Y = np.meshgrid(Xr, Yr)
Z = np.zeros((60, 101))

for i in range(101):
    data = df[df["fullmove"] == i]
    for j in range(60):
        Z[j, i] = data[data["legal_moves"] == j]["count"].sum()

ax = plt.pcolor(
    X, Y, Z,
    cmap="magma",
)

def ecdf(fullmove: int, threshold: float = 0):
    sub_df = df[df["fullmove"] == fullmove]
    samples = []
    for i in range(60):
        count = sub_df[sub_df["legal_moves"] == i]["count"].sum()
        samples.extend([i] * count)

    if len(samples) == 0:
        return np.nan

    # ecdf
    return np.quantile(samples, threshold)
   

for (label, M, color) in [
    ("$p=0.75$", 0.75, 'red'),
    ("$p=0.50$", 0.5, 'green'),
    ("$p=0.25$", 0.25, 'blue'),
    ("$p=0.00$", 0, 'orange'),
]:
    Y = np.array([ecdf(x, M) for x in Xr])
    plt.plot(Xr, Y, label=label, color=color, linestyle='dashed')

plt.legend(facecolor='white', framealpha=1, loc='upper right', title='Quantile')

plt.xlabel("Turn number (full moves)")
plt.ylabel("Number of available moves")
plt.title("Number of available moves for White throughout the game")

plt.tight_layout()
plt.savefig("./output/avg_moves.pdf", format='pdf')


#for p in [0.75, 0.5, 0.25, 0]:
#    print(p, [(x, ecdf(x, p)) for x in range(101)])
