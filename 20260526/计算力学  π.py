import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. 参数设置与计算
# =====================
# 定义边数 n (2的幂次，从1到256)
n_list = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

# 真实的圆周率
pi_true = np.pi

# 计算直接逼近的 π_n = n * sin(π/n)
pi_approx = n_list * np.sin(np.pi / n_list)

# 计算原始误差 e_n = |π - π_n|
error_original = np.abs(pi_true - pi_approx)

# =====================
# 2. 计算外推后的误差
# =====================
# 使用 Richardson 外推法消除主导的 h^2 误差项
# 公式: π_ext = (4 * π_{2n} - π_n) / 3
pi_extrapolated = []
n_extrapolated = []
error_extrapolated = []

for i in range(1, len(n_list)):
    # 利用 n 和 2n 的结果进行外推
    val_extrap = (4 * pi_approx[i] - pi_approx[i-1]) / 3
    pi_extrapolated.append(val_extrap)
    n_extrapolated.append(n_list[i])
    error_extrapolated.append(np.abs(pi_true - val_extrap))

# 转为 numpy 数组方便绘图
n_extrapolated = np.array(n_extrapolated)
error_extrapolated = np.array(error_extrapolated)

# =====================
# 3. 绘制双对数误差图
# =====================
plt.figure(figsize=(10, 6))

# 网格尺寸 h = 1/n
h_original = 1.0 / n_list
h_extrap = 1.0 / n_extrapolated

# A. 绘制原始误差 (蓝色实心圆 + 实线)
plt.loglog(h_original, error_original, 'o-', color='blue',
           label='Original Error (n=1 to 256)', markersize=8)

# B. 绘制外推误差 (红色实心三角形 + 实线)
plt.loglog(h_extrap, error_extrapolated, '^-', color='red',
           label='Extrapolation (Wynn-ε)', markersize=8)

# C. 对原始误差做线性拟合, 计算斜率, 并画出拟合线
# 选取 n>=16 的点进行拟合，避免极低阶时数值波动的影响
fit_start_idx = 4  # 对应 n=16
log_h_fit = np.log(h_original[fit_start_idx:])
log_e_fit = np.log(error_original[fit_start_idx:])
coeffs = np.polyfit(log_h_fit, log_e_fit, 1)
slope = coeffs[0]
intercept = coeffs[1]

# 画出拟合线 (延伸到全图范围以展示趋势)
fit_line = np.exp(intercept) * (h_original**slope)
plt.loglog(h_original, fit_line, 'r--',
           label=f'Fit line (slope ≈ {slope:.3f})', linewidth=2)

# =====================
# 4. 图表美化与标注
# =====================
plt.xlabel('h = 1/n', fontsize=14)
plt.ylabel('Error $e_n = |\pi - \pi_n|$', fontsize=14)
plt.title('Error Convergence in Log-Log Scale', fontsize=16)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# =====================
# 5. 打印计算结果 (控制台输出)
# =====================
print("--- 直接逼近结果 (原始) ---")
print(f"{'n':<6} {'π_n (raw)':<16} {'Error':<16}")
print("-" * 42)
for n, p, e in zip(n_list, pi_approx, error_original):
    print(f"{int(n):<6} {p:.12f} {e:.4e}")

print("\n--- 外推法结果 (Extrapolation) ---")
print(f"{'n (base)':<8} {'π_extrap':<16} {'Error after Extrap':<16}")
print("-" * 45)
for i in range(1, len(n_list)):
    base_n = n_list[i-1]  # 用于外推的较低阶 n
    final_n = n_list[i]   # 用于外推的较高阶 n
    print(f"{int(base_n)} & {int(final_n)}  {pi_extrapolated[i-1]:.12f} {error_extrapolated[i-1]:.4e}")

# =====================
# 6. 显示图形
# =====================
plt.show()