from Plots_Code.Data_Plots import *

probs_single = probs_list[0]

#%% Predictive Entropy Histogram nebeneinander (Single vs DE)
# Entropie berechnen
entropy_single = entropy(probs_single.T)
probs_array = np.array(probs_list)
mean_probs = np.mean(probs_array, axis=0)
entropy_ensemble = entropy(mean_probs.T)

# Binning
bins = np.linspace(0, np.max([entropy_single.max(), entropy_ensemble.max()]), 30)
bin_centers = (bins[:-1] + bins[1:]) / 2
bar_width = (bin_centers[1] - bin_centers[0]) * 0.45

# Histogramm zählen
counts_single, _ = np.histogram(entropy_single, bins=bins)
counts_ensemble, _ = np.histogram(entropy_ensemble, bins=bins)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(bin_centers - bar_width/2, counts_single, width=bar_width, label="Single BNN", color="r", alpha=0.7)
ax.bar(bin_centers + bar_width/2, counts_ensemble, width=bar_width, label="DE-BNN", alpha=0.7)

ax.set_xlabel("Predictive Entropy")
ax.set_ylabel("Frequency")
ax.set_title("Uncertainty Comparison (Predictive Entropy)")
ax.legend()
apply_custom_theme(ax)
fig.tight_layout()
plt.show()
