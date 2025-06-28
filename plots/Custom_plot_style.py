def apply_custom_theme(ax):
 xticks = ax.get_xticks()
 yticks = ax.get_yticks()
 ymid = [(yticks[i] + yticks[i+1]) / 2 for i in range(len(yticks) - 1)]
 ax.set_yticks(sorted(list(yticks) + ymid))

 for y in yticks:
  ax.axhline(y=y, color='gray', linewidth=0.4, linestyle='-', alpha=0.6)
 for x in xticks:
  ax.axvline(x=x, color='gray', linewidth=0.4, linestyle='-', alpha=0.6)

 for i in range(len(yticks) - 1):
  y_mid = (yticks[i] + yticks[i + 1]) / 2
  ax.axhline(y=y_mid, color='gray', linewidth=0.2, linestyle='-', alpha=0.2)
 for i in range(len(xticks) - 1):
  x_mid = (xticks[i] + xticks[i + 1]) / 2
  ax.axvline(x=x_mid, color='gray', linewidth=0.2, linestyle='-', alpha=0.2)

 ax.set_xlabel(ax.get_xlabel() or "x-axis", fontsize=32, labelpad=20)
 ax.set_ylabel(ax.get_ylabel() or "y-axis", fontsize=32, labelpad=20)
 ax.tick_params(axis='x', labelsize=22, pad=5)
 ax.tick_params(axis='y', labelsize=22, pad=10)
 ax.set_title(ax.get_title() or "Title", fontsize=28, fontweight='bold', color='black', loc='center', pad=15)
 ax.set_facecolor("white")

 for spine in ax.spines.values():
  spine.set_linewidth(1.0)
  spine.set_color("black")
 all_yticks = ax.get_yticks()
 ax.set_yticklabels([
  ax.yaxis.get_major_formatter().format_data(tick) if i % 2 == 0 else ""
  for i, tick in enumerate(all_yticks)
 ])

