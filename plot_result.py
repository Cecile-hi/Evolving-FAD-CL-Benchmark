import matplotlib.pyplot as plt
import numpy as np

# 时间轴
time = np.linspace(0, 10, 400)

# 第一条曲线，快速下降后平稳
curve1 = np.exp(-time) * 10 - 2

# 第二条曲线，慢速下降后平稳
curve2 = np.exp(-0.5 * time) * 10

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(time, curve1, label='Ours')
plt.plot(time, curve2, label='Base LoRA')
plt.xlabel('epoch')
plt.ylabel('EER')
plt.title('The Comprison of two methods')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("compare_result.jpg")
