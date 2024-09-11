import seaborn as sns

def make_heatmap(
    ax,
    df,
    x_col, x_label,
    y_col, y_label,
    value_col, value_label,
    cmap = "flare",
    balance_cmap=False,
    right_ticks=False
):
    if balance_cmap:
        min_val = df[value_col].min()
        max_val = df[value_col].max()
        max_abs = max(abs(min_val), abs(max_val))

    sns.heatmap(
        df.pivot(index=y_col, columns=x_col, values=value_col),
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=False,
        cbar_kws={'label': value_label},
        ax=ax,
        vmin=(-max_abs) if balance_cmap else df[value_col].min(),
        vmax=max_abs if balance_cmap else df[value_col].max()
    )
    ax.set_title(value_label)
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    if right_ticks:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    
    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(fs(sub_df["feature-set"].iloc[0]))
    ax.tick_params(axis='y', labelrotation=0)
