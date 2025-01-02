import matplotlib.pyplot as plt
import numpy as np

data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_accuracies = [np.random.rand(9) * 100, 
                   [0.010981225646475381, 0.010361317747077577, 0.010323364202216486, 0.010272759475735034, 0.011052072263549416, 0.013018065887353878, 0.011335458731845554, 0.013283740701381509 , 0.018865887353878853], 
                   [7.061046168378793, 100, 100, 100, 100, 100, 100, 100, 100], 
                   np.random.rand(9) * 100, 
                   np.random.rand(9) * 100, 
                   [1.0745070256228598, 1.726886291179596, 7.195992105662669, 99.8051718030464, 100, 100, 100, 100, 100]]

titles = ["SGD", "Adam", "AdamW, weight decay 1", "SGD with momentum", "Adam, 0.3x baseline LR", "Dropout 0.1, AdamW"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    ax.scatter(data_fractions, best_accuracies[i], alpha=0.5)
    ax.plot(data_fractions, best_accuracies[i], color='b')
    ax.set_title(titles[i])
    ax.set_ylim(-3, 103)

fig.text(0.55, 0.04, 'Training data fraction', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Best validation accuracy', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()

