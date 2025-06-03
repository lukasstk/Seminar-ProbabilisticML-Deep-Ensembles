import matplotlib.pyplot as plt


def apply_custom_theme(ax):
 # x-Achsentitel (oben Abstand t=20, Größe 32)
 ax.set_xlabel(ax.get_xlabel() or "x-axis", fontsize=32, labelpad=20)

 # x-Ticks (Größe 22, Abstand t=5, zentriert)
 ax.tick_params(axis='x', labelsize=22, pad=5)

 # y-Achsentitel (rechts Abstand r=20, Größe 32)
 ax.set_ylabel(ax.get_ylabel() or "y-axis", fontsize=32, labelpad=20)

 # y-Ticks (Größe 22, Abstand r=10)
 ax.tick_params(axis='y', labelsize=22, pad=10)

 # Titel (Größe 28, zentriert, bold, schwarz)
 ax.set_title(ax.get_title() or "Title", fontsize=28, fontweight='bold', color='black', loc='center', pad=15)

 # Haupt-Gitterlinien
 ax.grid(True, which='major', linestyle='-', linewidth=0.2, color='darkgray')

 # Neben-Gitterlinien
 ax.minorticks_on()
 ax.grid(True, which='minor', linestyle='-', linewidth=0.1, color='gray')

 # Hintergrundfarbe (Panel)
 ax.set_facecolor("white")

 # Achsenlinien (Spines)
 for spine in ax.spines.values():
  spine.set_linewidth(1.0)
  spine.set_color("black")

