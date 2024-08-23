import numpy as np
import matplotlib.pyplot as plt

# 设置初始参数
epochs = 200
initial_loss = 1.0  # 初始损失值
final_loss = 0.01   # 最终损失值，趋近于0但不是0

# 使用指数衰减函数生成损失数据
# y = (initial_loss - final_loss) * exp(-k * x) + final_loss
# 其中 k 是衰减率，x 是 epoch 数，y 是损失值
# 我们可以通过调整 k 来控制损失值减少的快慢

# 计算衰减率 k
# 我们希望在前10%的 epoch 中损失值减少得快，后90%的 epoch 中减少得慢
decay_rate_fast = np.log((final_loss / initial_loss) / 0.1) / (0.1 * epochs)
decay_rate_slow = np.log((final_loss / initial_loss) / 0.9) / (0.9 * epochs)

loss_values = []

for epoch in range(1, epochs + 1):
    if epoch <= 0.1 * epochs:
        k = decay_rate_fast
    else:
        k = decay_rate_slow
    loss = (initial_loss - final_loss) * np.exp(-k * epoch) + final_loss
    loss_values.append(loss)

# 将列表转换为 numpy 数组以便于绘图
loss_values = np.array(loss_values)

# 绘制损失曲线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), loss_values, label='Training Loss')
plt.title('Training Loss Curve with Exponential Decay')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()