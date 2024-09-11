import seaborn as sns

def make_heatmap(
    ax,
    df,
    x_col, x_label,
    y_col, y_label,
    value_col, value_label,
    cmap = "flare",
    decimals=4,
    balance_cmap=False,
    right_ticks=False
):
    sns.heatmap(
        df.pivot(index=y_col, columns=x_col, values=value_col),
        annot=True,
        fmt=f".{decimals}f",
        cmap=cmap,
        cbar=False,
        cbar_kws={'label': value_label},
        ax=ax,
        vmin=df[value_col].min(),
        vmax=df[value_col].max()
    )
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    if right_ticks:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    
    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')
    ax.set_title(value_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='y', labelrotation=0)
