import matplotlib.pyplot as plt
import numpy as np

# 示例数据
data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_accuracies = [np.random.rand(9) * 100, 
                   np.random.rand(9) * 100, 
                   [7.061046168378793, 100, 100, 100, 100, 100, 100, 100, 100], 
                   np.random.rand(9) * 100, 
                   np.random.rand(9) * 100, 
                   np.random.rand(9) * 100]

# 图表标题
titles = ["SGD", "Adam", "AdamW, weight decay 1", "SGD with momentum", "Adam, 0.3x baseline LR", "Dropout 0.1, AdamW"]

# 创建图形和子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 绘制每个子图
for i, ax in enumerate(axes.flat):
    ax.scatter(data_fractions, best_accuracies[i], alpha=0.5)
    ax.plot(data_fractions, best_accuracies[i], color='b')
    ax.set_title(titles[i])
    ax.set_ylim(-3, 103)

# 设置全局的 xlabel 和 ylabel
fig.text(0.55, 0.04, 'Training data fraction', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Best validation accuracy', va='center', rotation='vertical', fontsize=12)

# 调整布局
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()

