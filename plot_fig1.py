import matplotlib.pyplot as plt

# Transformer architecture
data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_epochs_transformer = [1e5, 2160, 605, 291, 127, 86, 59, 41, 37]

plt.figure(figsize=(10, 6))
plt.plot(data_fractions, num_epochs_transformer, label='transformer', linewidth=2)
plt.scatter(data_fractions[0], num_epochs_transformer[0], color='r', marker='^', s=100, label="Runs that didn't reach 99% val acc in $10^5$ updates")
plt.yscale('log')
plt.xlabel('Training data fraction')
plt.ylabel('Steps to validation accuracy > 99%')
plt.title('Steps until generalization for modular sum')
plt.legend()
plt.grid(True)

plt.show()