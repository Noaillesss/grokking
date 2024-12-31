import matplotlib.pyplot as plt

# Transformer architecture
data_fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
num_epochs_transformer = [5e5, 23271, 7840, 4126, 2022, 500, 158]

plt.figure(figsize=(10, 6))
plt.plot(data_fractions, num_epochs_transformer, label='mlp', linewidth=2)
plt.scatter(data_fractions[0], num_epochs_transformer[0], color='r', marker='^', s=100, label="Runs that didn't reach 99% val acc in $5×10^5$ updates")
plt.yscale('log')
plt.xlabel('Training data fraction')
plt.ylabel('Steps to validation accuracy > 99%')
plt.title('Steps until generalization for modular sum')
plt.legend()
plt.grid(True)

plt.show()