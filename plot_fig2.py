import matplotlib.pyplot as plt
import numpy as np

data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_accuracies = [[None, 1.9394261424017, 3.9775315014422344, 10.18420120439249, 89.41551540913921, 99.94686503719448, 100.0, 100.0, None], # SGD
                   [2.0073208170976502, 100, 100, 100, 100, 100, 100, 100, 100], # weight decay = 0.0001
                   [7.061046168378793, 100, 100, 100, 100, 100, 100, 100, 100], 
                   [None, 1.1955366631243358, 2.3531197813875818, 7.208643287283033, 63.50690754516471, 97.66206163655685, 99.89373007438895, 99.84059511158342, None], # SGD with momentum
                   [1.1335458731845554, 1.3018065887353878, 2.3986640352208898, 97.75061990789939, 100, 100, 100, 100, 100], 
                   [1.0745070256228598, 1.726886291179596, 7.195992105662669, 99.8051718030464, 100, 100, 100, 100, 100]]

titles = ["SGD", "Adam", "AdamW, weight decay 1", "SGD with momentum", "Adam, 0.1x baseline LR", "Dropout 0.1, AdamW"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    ax.scatter(data_fractions, best_accuracies[i], alpha=0.5)
    ax.plot(data_fractions, best_accuracies[i], color='b')
    ax.set_title(titles[i])
    ax.set_ylim(-3, 103)

fig.text(0.55, 0.04, 'Training data fraction', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Best validation accuracy', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])

plt.savefig('figures/best_acc.eps', format='eps')
plt.show()

