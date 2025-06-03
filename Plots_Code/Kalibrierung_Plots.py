from Plots_Code.Data_Plots import *
from imports import *

probs_single = probs_list[0]

# Prepare data
y_pred_s = np.argmax(probs_single, axis=1)
conf_s = np.max(probs_single, axis=1)
correct_s = (y_pred_s == y_test).astype(int)
prob_true_s, prob_pred_s = calibration_curve(correct_s, conf_s, n_bins=10)

ensemble_probs = np.mean(probs_list, axis=0)
y_pred_e = np.argmax(ensemble_probs, axis=1)
conf_e = np.max(ensemble_probs, axis=1)
correct_e = (y_pred_e == y_test).astype(int)
prob_true_e, prob_pred_e = calibration_curve(correct_e, conf_e, n_bins=10)

# Compute ECE
def compute_ece(y_true, y_probs, n_bins=10):
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(int)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_bounds[i], bin_bounds[i+1]
        mask = (confidences >= bin_lower) & (confidences < bin_upper)
        if np.any(mask):
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            ece += np.abs(bin_conf - bin_acc) * np.mean(mask)
    return ece

ece_single = compute_ece(y_test, probs_single)
ece_ensemble = compute_ece(y_test, ensemble_probs)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred_e, prob_true_e, marker='o', label="Ensemble")
ax.plot(prob_pred_s, prob_true_s, marker='o', linestyle='--', label="Single", color='r')
ax.plot([0, 1], [0, 1], linestyle=':', color='gray')
ax.set_xlabel("Confidence")
ax.set_ylabel("Accuracy")
ax.set_title("Reliability Diagram")
ax.legend()
apply_custom_theme(ax)

fig.tight_layout()
fig.suptitle("Model Calibration (ECE↓)\nSingle: {:.4f} | Ensemble: {:.4f}".format(ece_single, ece_ensemble),
              fontsize=16, y=1.08)

plt.show()

# Konfidenz pro Sample (max. Softmax)
conf_single = np.max(probs_single, axis=1)
conf_ensemble = np.max(np.mean(probs_list, axis=0), axis=1)

# Accuracy pro Sample
acc_single = (np.argmax(probs_single, axis=1) == y_test).astype(int)
acc_ensemble = (np.argmax(np.mean(probs_list, axis=0), axis=1) == y_test).astype(int)

# Confidence Bins
bins = np.linspace(0, 1, 11)  # 10 bins: [0.0–0.1), ..., [0.9–1.0]

# Bin indices
bin_idx_s = np.digitize(conf_single, bins) - 1
bin_idx_e = np.digitize(conf_ensemble, bins) - 1

# Correct bin index range: only 10 bins
num_bins = len(bins) - 1  # 10

acc_bin_s = [acc_single[bin_idx_s == i].mean() if np.any(bin_idx_s == i) else np.nan for i in range(num_bins)]
acc_bin_e = [acc_ensemble[bin_idx_e == i].mean() if np.any(bin_idx_e == i) else np.nan for i in range(num_bins)]

bin_centers = (bins[:-1] + bins[1:]) / 2  # shape (10,)

#%% Confidence Histogram (einzeln)
bar_width = 0.04
counts_single, _ = np.histogram(conf_single, bins=bins)
counts_ensemble, _ = np.histogram(conf_ensemble, bins=bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(bin_centers - bar_width/2, counts_single, width=bar_width, label="Single", color="r", alpha=0.7)
ax1.bar(bin_centers + bar_width/2, counts_ensemble, width=bar_width, label="Ensemble", alpha=0.7)
ax1.set_xlabel("Confidence")
ax1.set_ylabel("Frequency")
ax1.set_title("Prediction Confidence Histogram")
ax1.legend()
apply_custom_theme(ax1)
fig1.tight_layout()
plt.show()

#%% Accuracy vs Confidence (einzeln)
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(bin_centers, acc_bin_e, marker='o', label="Ensemble")
ax2.plot(bin_centers, acc_bin_s, marker='o', linestyle='--', label="Single", color='r')
ax2.set_xlabel("Confidence bin")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy vs Confidence")
ax2.set_ylim(0, 1)
ax2.legend()
apply_custom_theme(ax2)
fig2.tight_layout()
plt.show()