import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = "\n".join([
    r'\usepackage{skak}',
    r'\usepackage{xskak}'
])

df = pd.read_csv('../../assets/results/mobility.csv')

print(df.describe())

# remove value=0 for some categories
df = df[
    (df["value"] != 0) | (
        (df["piece"] == "Pawn")|
        (df["piece"] == "King")
    )
]

# g = sns.catplot(df, x="value", y="count", col="piece", kind="bar", col_wrap=3, sharex=False, sharey=False)
# g.set_titles("{col_name} pieces")

def get_piece_char(piece):
    if piece == "Pawn":
        return "\\sympawn"
    if piece == "Knight":
        return "\\symknight"
    if piece == "Bishop":
        return "\\symbishop"
    if piece == "Rook":
        return "\\symrook"
    if piece == "Queen":
        return "\\symqueen"
    if piece == "King":
        return "\\symking"

def annotate(data, **kws):
    piece = data['piece'].iloc[0]
    n = len(data)
    ax = plt.gca()
    ax.text(.7, .5, get_piece_char(piece), fontsize=60, color="gray", transform=ax.transAxes)
    ax.set_title(f"{piece} mobility")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # set bar colors


    right = None


    if piece == "Pawn":
        right = 9 + 0.5
    if piece == "Knight":
        right = 15 + 0.5
    if piece == "Bishop":
        right = 20 + 0.5
    if piece == "Rook":
        right = 26 + 0.5
    if piece == "Queen":
        right = 26 + 0.5

    ax.set_xlim(None, right)

g = sns.FacetGrid(df, col="piece", col_wrap=3, sharex=False, sharey=False, aspect=1.2)
g.map_dataframe(sns.barplot, x="value", y="count", errorbar=('ci', False), color=sns.color_palette()[1])
g.map_dataframe(annotate)


plt.tight_layout()
plt.savefig("./output/mobility.pdf", format='pdf')
