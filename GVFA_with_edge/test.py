import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("final_data/solubility_1.csv")
test_df = pd.read_csv("final_data/testset_novel.csv")

train_logS = train_df["logS"].dropna()
test_logS = test_df["logS"].dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(train_logS, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
ax1.axvline(train_logS.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {train_logS.mean():.2f}")
ax1.set_xlabel("logS (log mol/L)", fontsize=13)
ax1.set_ylabel("Count", fontsize=13)
ax1.set_title(f"Training Set (n={len(train_logS)})", fontsize=14)
ax1.legend(fontsize=11)
ax1.tick_params(labelsize=11)

ax2.hist(test_logS, bins=20, color="#DD8452", edgecolor="white", alpha=0.8)
ax2.axvline(test_logS.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {test_logS.mean():.2f}")
ax2.set_xlabel("logS (log mol/L)", fontsize=13)
ax2.set_ylabel("Count", fontsize=13)
ax2.set_title(f"Novel Test Set (n={len(test_logS)})", fontsize=14)
ax2.legend(fontsize=11)
ax2.tick_params(labelsize=11)

fig.suptitle("Distribution of logS", fontsize=15, y=1.02)
fig.tight_layout()
plt.savefig("logS_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Train: n={len(train_logS)}, mean={train_logS.mean():.3f}, "
      f"std={train_logS.std():.3f}, range=[{train_logS.min():.2f}, {train_logS.max():.2f}]")
print(f"Test:  n={len(test_logS)}, mean={test_logS.mean():.3f}, "
      f"std={test_logS.std():.3f}, range=[{test_logS.min():.2f}, {test_logS.max():.2f}]")
