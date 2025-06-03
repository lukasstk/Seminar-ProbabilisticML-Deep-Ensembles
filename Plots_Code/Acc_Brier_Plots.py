from Plots_Code.Data_Plots import *
from imports import *

#%% Plots Accuracy and Brier Score
# Evaluate single model (first)
probs_single = probs_list[0]
preds_single = np.argmax(probs_single, axis=1)
acc_single = accuracy_score(y_test, preds_single)
brier_single = np.mean([
    brier_score_loss((y_test == i).astype(int), probs_single[:, i])
    for i in range(probs_single.shape[1])
])

# Evaluate ensembles of size 1 to ensemble_size
ensemble_sizes = list(range(1, ensemble_size + 1))
accs = []
briers = []

for k in ensemble_sizes:
    partial_probs = np.mean(probs_list[:k], axis=0)
    preds = np.argmax(partial_probs, axis=1)
    accs.append(accuracy_score(y_test, preds))
    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), partial_probs[:, i])
        for i in range(partial_probs.shape[1])
    ])
    briers.append(brier)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.plot(ensemble_sizes, accs, label="Ensemble", marker='o')
ax1.axhline(y=acc_single, color='r', linestyle='--', label="Single")
ax1.set_xticks(range(1, ensemble_size + 1))
ax1.set_xlabel("Ensemble size")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("Accuracy")
ax1.legend()
apply_custom_theme(ax1)

# Brier score plot
ax2.plot(ensemble_sizes, briers, label="Ensemble", marker='o')
ax2.axhline(y=brier_single, color='r', linestyle='--', label="Single")
ax2.set_xticks(range(1, ensemble_size + 1))
ax2.set_xlabel("Ensemble size")
ax2.set_ylabel("Brier Score")
ax2.set_title("Brier Score")
ax2.legend()
apply_custom_theme(ax2)

# Layout and super title
fig.tight_layout()
fig.suptitle("ConvolutionalBNN Ensemble on CIFAR-10", fontsize=18, y=1.05)

plt.show()


#%% Plots NLL and Predictive Entropy
from imports import *
# Calculate NLL and Predictive Entropy
nlls = []
entropies = []

for k in range(1, len(probs_list) + 1):
    partial_probs = np.mean(probs_list[:k], axis=0)
    nll = log_loss(y_test, partial_probs)
    ent = entropy(partial_probs.T)
    nlls.append(nll)
    entropies.append(np.mean(ent))

# Single model values
nll_single = log_loss(y_test, probs_single)
entropy_single = np.mean(entropy(probs_single.T))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# NLL plot
ax1.plot(ensemble_sizes, nlls, label="Ensemble", marker='o')
ax1.axhline(nll_single, color='red', linestyle='--', label="Single")
ax1.set_xticks(range(1, ensemble_size + 1))
ax1.set_xlabel("Ensemble size")
ax1.set_ylabel("Negative Log-Likelihood")
ax1.set_title("NLL vs Ensemble Size")
ax1.legend()
apply_custom_theme(ax1)

# Entropy plot
ax2.plot(ensemble_sizes, entropies, label="Ensemble", marker='o')
ax2.axhline(entropy_single, color='red', linestyle='--', label="Single")
ax2.set_xticks(range(1, ensemble_size + 1))
ax2.set_xlabel("Ensemble size")
ax2.set_ylabel("Predictive Entropy")
ax2.set_title("Entropy vs Ensemble Size")
ax2.legend()
apply_custom_theme(ax2)

fig.tight_layout()
fig.suptitle("Uncertainty Evaluation of ConvolutionalBNN Ensemble", fontsize=18, y=1.05)
plt.show()