import matplotlib.pyplot as plt
import torch


# plotting
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def style_plot(ax, M, title, legend=False):
    # Axes styling
    ax.set_xlim(-M, M)
    ax.set_ylim(-M, M)

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Keep box/spines visible
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # No grid
    ax.grid(False)

    # Title
    ax.set_title(
        title,
        fontsize=18,
        fontfamily='CMU Sans Serif',
        fontweight='bold'
    )

    # Legend
    if legend:
        legend = ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=13
        )
    else:
        legend = None

    return ax