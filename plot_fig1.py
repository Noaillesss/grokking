import matplotlib.pyplot as plt

# all model architecture
data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_epochs_transformer = [1e5, 2160, 605, 291, 127, 86, 59, 41, 37]
num_epochs_mlp = [1e5, 1e5, 23271, 7840, 4126, 2022, 500, 158]
num_epochs_lstm = [1e5, 14688, 4446, 986, 266, 251, 109, 85, 938]

scatter_x = [0.1, 0.2]
scatter_y = [1e5, 1e5]

plt.figure(figsize=(10, 6))
plt.plot(data_fractions, num_epochs_transformer, label='transformer', linewidth=2)
plt.plot(data_fractions[:8], num_epochs_mlp, c='#2ca02c', label='mlp', linewidth=2)
plt.plot(data_fractions, num_epochs_lstm, c='#9467bd', label='lstm', linewidth=2)
plt.scatter(scatter_x, scatter_y, color='r', marker='^', s=100, label="Runs that didn't reach 99% val acc in $10^5$ updates")
plt.yscale('log')
plt.xlabel('Training data fraction')
plt.ylabel('Steps to validation accuracy > 99%')
plt.title('Steps until generalization for modular sum')
plt.legend()
plt.grid(True)

plt.savefig('figures/architecture_acc.eps', format='eps')
plt.show()
