"""
有限元平衡方程组求解器（2.4 作业）
复用 2.3 作业的桁架装配模块，新增：
- 稠密 LDL^T 分解与求解
- 病态矩阵误差分析
- 稀疏矩阵生成与 scipy.sparse 求解（模拟大规模求解器）
- 二维 Poisson 方程有限元算例（Q4 单元）
"""

import numpy as np
import json
import sys
import time
import math
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt  # 可选，用于绘图

# ==================== 2.3 复用模块（略作调整） ====================

def read_input_from_dict(data):
    """与 2.3 相同，从字典读取模型数据"""
    nsd = data['nsd']; ndof = data['ndof']; nnp = data['nnp']; nel = data['nel']; nen = data['nen']
    E = np.array(data['E']); A = np.array(data['CArea'])
    x = np.array(data['x']); y = np.array(data['y']) if nsd == 2 else np.zeros(nnp)
    IEN = np.array(data['IEN']) - 1
    fixed_dof = np.array(data['fixed_dof']) - 1
    fixed_val = np.array(data['fixed_value'])
    force_dof = np.array(data['force_dof']) - 1
    force_val = np.array(data['force_value'])
    return nsd, ndof, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, force_dof, force_val

def generate_LM(IEN, ndof):
    nel, nen = IEN.shape
    LM = np.zeros((nel, nen*ndof), dtype=int)
    for e in range(nel):
        for a in range(nen):
            node = IEN[e, a]
            for i in range(ndof):
                LM[e, a*ndof + i] = node * ndof + i
    return LM

def element_stiffness_1D(E, A, L):
    return (E * A / L) * np.array([[1, -1], [-1, 1]])

def element_stiffness_2D(E, A, L, cos, sin):
    c, s = cos, sin
    k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
    T = np.array([[c, s, 0, 0], [0, 0, c, s]])
    return T.T @ k_local @ T

def assemble_global_stiffness(nnp, ndof, nel, nen, LM, IEN, E, A, x, y):
    N = nnp * ndof
    K = np.zeros((N, N))
    for e in range(nel):
        node1, node2 = IEN[e, 0], IEN[e, 1]
        L = math.hypot(x[node2]-x[node1], y[node2]-y[node1])
        if ndof == 1:
            Ke = element_stiffness_1D(E[e], A[e], L)
        else:
            dx = x[node2] - x[node1]; dy = y[node2] - y[node1]
            cos = dx / L; sin = dy / L
            Ke = element_stiffness_2D(E[e], A[e], L, cos, sin)
        dof_list = LM[e]
        for i in range(len(dof_list)):
            for j in range(len(dof_list)):
                K[dof_list[i], dof_list[j]] += Ke[i, j]
    return K

def reduce_system(K, fixed_dof, fixed_val, force_dof, force_val):
    """生成缩减方程 K_FF d_F = rhs，返回 free_dof 和 rhs"""
    N = K.shape[0]
    all_dof = np.arange(N)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    K_FF = K[np.ix_(free_dof, free_dof)]
    K_FE = K[np.ix_(free_dof, fixed_dof)]
    F = np.zeros(N)
    F[force_dof] = force_val
    F_F = F[free_dof]
    rhs = F_F - K_FE @ fixed_val
    return K_FF, rhs, free_dof

def postprocess_1D(nel, IEN, E, A, x, d, ndof):
    results = []
    for e in range(nel):
        node1, node2 = IEN[e,0], IEN[e,1]
        L = x[node2] - x[node1]
        u1, u2 = d[node1*ndof], d[node2*ndof]
        de = np.array([u1, u2])
        sigma = E[e] / L * np.array([-1, 1]) @ de
        results.append({'element': e+1, 'length': L, 'stress': sigma, 'axial_force': sigma * A[e]})
    return results

def postprocess_2D(nel, IEN, E, A, x, y, d, ndof):
    results = []
    for e in range(nel):
        node1, node2 = IEN[e,0], IEN[e,1]
        dx = x[node2] - x[node1]; dy = y[node2] - y[node1]
        L = math.hypot(dx, dy)
        cos = dx / L; sin = dy / L
        dof_idx = [node1*ndof, node1*ndof+1, node2*ndof, node2*ndof+1]
        de = d[dof_idx]
        sigma = E[e] / L * np.array([-cos, -sin, cos, sin]) @ de
        results.append({'element': e+1, 'length': L, 'cos': cos, 'sin': sin,
                        'stress': sigma, 'axial_force': sigma * A[e]})
    return results

# ==================== 2.4 新增：LDL^T 求解器 ====================

def ldlt_factor(K):
    """
    对对称矩阵 K 进行 LDL^T 分解（无选主元）
    返回 L（单位下三角）和 D（对角元）
    若出现非正主元，抛出 ValueError
    """
    n = K.shape[0]
    # 复制矩阵，原地修改
    A = K.astype(float).copy()
    L = np.eye(n)
    D = np.zeros(n)
    for j in range(n):
        # 计算 D[j]
        d = A[j, j]
        for k in range(j):
            d -= L[j, k]**2 * D[k]
        if d <= 1e-15:
            raise ValueError(f"LDLT 分解失败：第 {j} 个主元 {d} 非正（可能矩阵非正定或奇异）")
        D[j] = d
        # 更新 L 的后续列
        for i in range(j+1, n):
            val = A[i, j]
            for k in range(j):
                val -= L[i, k] * L[j, k] * D[k]
            L[i, j] = val / D[j]
    return L, D

def ldlt_solve(L, D, rhs):
    """
    求解 L D L^T x = rhs
    步骤：前代 (L y = rhs) -> 对角 (D z = y) -> 回代 (L^T x = z)
    """
    n = L.shape[0]
    # 前代
    y = np.zeros(n)
    for i in range(n):
        s = rhs[i] - np.dot(L[i, :i], y[:i])
        y[i] = s
    # 对角
    z = y / D
    # 回代（解 L^T x = z，注意 L^T 是上三角）
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = z[i] - np.dot(L[i+1:, i], x[i+1:])  # 注意 L 是下三角，L[i+1:, i] 是列向量
        x[i] = s
    return x

def residual_norm(K, x, rhs):
    r = rhs - K @ x
    return r, np.linalg.norm(r)

def condition_number_2(K):
    """计算 2-范数条件数（仅用于分析）"""
    # 注意：对于大型矩阵，计算 SVD 可能很耗时，此处仅用于小规模演示
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-12]  # 过滤零特征值
    if len(eigvals) == 0:
        return np.inf
    return np.max(eigvals) / np.min(eigvals)

# ==================== 2.4 任务2：病态矩阵分析 ====================

def ill_conditioned_test():
    """算例2：病态矩阵误差分析"""
    print("\n" + "="*60)
    print("病态矩阵测试")
    print("="*60)
    K = np.array([[1.0000, 1.0000], [1.0000, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    print("K =", K)
    print("精确解 a_exact =", a_exact)
    print("R =", R)
    print("条件数 cond(K) =", condition_number_2(K))

    # 双精度求解
    L, D = ldlt_factor(K)
    a_double = ldlt_solve(L, D, R)
    r_double, nr_double = residual_norm(K, a_double, R)
    err_double = np.linalg.norm(a_double - a_exact) / np.linalg.norm(a_exact)

    # 模拟 4 位有效数字（截断）
    def truncate_to_4sig(x):
        if abs(x) < 1e-12:
            return 0.0
        # 保留4位有效数字（四舍五入）
        return round(x, 3 - int(math.floor(math.log10(abs(x)))))
    K4 = np.vectorize(truncate_to_4sig)(K)
    R4 = np.vectorize(truncate_to_4sig)(R)
    print("\n截断后的 K4 =", K4)
    try:
        L4, D4 = ldlt_factor(K4)
        a_4sig = ldlt_solve(L4, D4, R4)
        r_4sig, nr_4sig = residual_norm(K, a_4sig, R)  # 残差用原 K 计算
        err_4sig = np.linalg.norm(a_4sig - a_exact) / np.linalg.norm(a_exact)
        print("\n--- 4位有效数字结果 ---")
        print(f"解 a = {a_4sig}")
        print(f"残差范数 ||r|| = {nr_4sig:.2e}")
        print(f"相对残差 ||r||/||R|| = {nr_4sig/np.linalg.norm(R):.2e}")
        print(f"相对误差 ||a-a_exact||/||a_exact|| = {err_4sig:.2e}")
    except ValueError as e:
        print("\n--- 4位有效数字结果 ---")
        print(f"LDLT 分解失败: {e}")
        print("这说明截断后的矩阵已变为奇异（或非正定），无法进行 LDLT 分解。")
        print("这也说明病态矩阵对数值误差非常敏感，微小的截断可能导致奇异性。")

    print("\n--- 双精度结果 ---")
    print(f"解 a = {a_double}")
    print(f"残差范数 ||r|| = {nr_double:.2e}")
    print(f"相对残差 ||r||/||R|| = {nr_double/np.linalg.norm(R):.2e}")
    print(f"相对误差 ||a-a_exact||/||a_exact|| = {err_double:.2e}")

    print("\n结论：病态矩阵即使残差很小，解也可能严重偏离真解（相对误差大）。")

# ==================== 2.4 任务3 & 算例4：Poisson 方程有限元求解 ====================

def build_poisson_q4(nx, ny):
    """
    生成单位正方形 [0,1]x[0,1] 上的 Q4 单元网格
    返回：节点坐标 (nnp,2)，单元连接 IEN (nel,4)，边界自由度标记
    """
    nnp = (nx+1)*(ny+1)
    nel = nx*ny
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    coords = np.array([[xi, yi] for yi in y for xi in x])
    # 单元连接 (四个节点：左下、右下、右上、左上)
    IEN = []
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1) + i
            n1 = n0 + 1
            n2 = n1 + (nx+1)
            n3 = n0 + (nx+1)
            IEN.append([n0, n1, n2, n3])
    IEN = np.array(IEN)
    # 边界节点（x=0 或 x=1 或 y=0 或 y=1）
    eps = 1e-12
    boundary = []
    for idx, (xi, yi) in enumerate(coords):
        if abs(xi) < eps or abs(xi-1) < eps or abs(yi) < eps or abs(yi-1) < eps:
            boundary.append(idx)
    boundary = np.array(boundary)
    return coords, IEN, boundary

def assemble_poisson_q4(coords, IEN):
    """
    装配 Poisson 方程 -Δu = f 的有限元刚度矩阵和载荷向量（Q4 单元，数值积分 2x2 Gauss）
    f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    """
    nnp = coords.shape[0]
    nel = IEN.shape[0]
    K = lil_matrix((nnp, nnp))
    R = np.zeros(nnp)
    # Gauss 积分点及权重 (2x2)
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3), -1/np.sqrt(3)],
                          [ 1/np.sqrt(3),  1/np.sqrt(3)],
                          [-1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    # 形函数在局部坐标 (ξ,η) 的值和导数
    def N(ξ, η):
        return np.array([(1-ξ)*(1-η)/4, (1+ξ)*(1-η)/4,
                         (1+ξ)*(1+η)/4, (1-ξ)*(1+η)/4])
    def dN_dξ(ξ, η):
        return np.array([-(1-η)/4, (1-η)/4, (1+η)/4, -(1+η)/4])
    def dN_dη(ξ, η):
        return np.array([-(1-ξ)/4, -(1+ξ)/4, (1+ξ)/4, (1-ξ)/4])

    for e in range(nel):
        nodes = IEN[e]
        xe = coords[nodes, 0]
        ye = coords[nodes, 1]
        Ke = np.zeros((4,4))
        Re = np.zeros(4)
        for gp in range(4):
            ξ, η = gauss_pts[gp]
            w = weights[gp]
            # 形函数
            NN = N(ξ, η)
            dNdξ = dN_dξ(ξ, η)
            dNdη = dN_dη(ξ, η)
            # Jacobian
            J = np.array([[dNdξ @ xe, dNdη @ xe],
                          [dNdξ @ ye, dNdη @ ye]])
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            # 形函数对全局坐标的导数
            dNdx = invJ[0,0]*dNdξ + invJ[0,1]*dNdη
            dNdy = invJ[1,0]*dNdξ + invJ[1,1]*dNdη
            # 单元刚度矩阵
            for i in range(4):
                for j in range(4):
                    Ke[i,j] += w * (dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j]) * detJ
            # 载荷向量 (f = 2*pi^2*sin(pi*x)*sin(pi*y))
            xg = NN @ xe
            yg = NN @ ye
            f = 2 * np.pi**2 * np.sin(np.pi * xg) * np.sin(np.pi * yg)
            for i in range(4):
                Re[i] += w * NN[i] * f * detJ
        # 组装到总体矩阵
        for i in range(4):
            gi = nodes[i]
            R[gi] += Re[i]
            for j in range(4):
                gj = nodes[j]
                K[gi, gj] += Ke[i, j]
    return K, R

def solve_poisson(nx, ny, method='sparse'):
    """求解 Poisson 方程，返回数值解和误差"""
    print(f"\n--- Poisson 算例: nx={nx}, ny={ny} ---")
    coords, IEN, boundary = build_poisson_q4(nx, ny)
    nnp = coords.shape[0]
    print(f"节点数: {nnp}, 单元数: {len(IEN)}")

    # 装配
    t0 = time.time()
    K, R = assemble_poisson_q4(coords, IEN)
    ass_time = time.time() - t0

    # 处理 Dirichlet 边界条件（零位移）
    free_dofs = np.setdiff1d(np.arange(nnp), boundary)
    K_ff = K[np.ix_(free_dofs, free_dofs)].tocsc()  # 转为 CSC 以便求解
    R_f = R[free_dofs]

    # 求解
    t0 = time.time()
    if method == 'sparse':
        u_f = spsolve(K_ff, R_f)
    elif method == 'ldlt':
        # 将 K_ff 转为稠密，用 LDLT 求解（仅用于小规模对比）
        K_dense = K_ff.toarray()
        L, D = ldlt_factor(K_dense)
        u_f = ldlt_solve(L, D, R_f)
    else:
        raise ValueError("未知求解方法")
    solve_time = time.time() - t0

    # 重构完整解
    u = np.zeros(nnp)
    u[free_dofs] = u_f

    # 计算理论解和误差
    u_exact = np.sin(np.pi * coords[:,0]) * np.sin(np.pi * coords[:,1])
    err = u - u_exact
    max_err = np.max(np.abs(err))
    l2_rel_err = np.sqrt(np.sum(err**2) / np.sum(u_exact**2))
    # 残差
    r = R - K @ u
    rel_res = np.linalg.norm(r) / np.linalg.norm(R)

    print(f"求解器: {method}")
    print(f"装配时间: {ass_time:.4f}s, 求解时间: {solve_time:.4f}s")
    print(f"最大节点误差: {max_err:.4e}")
    print(f"离散 L2 相对误差: {l2_rel_err:.4e}")
    print(f"相对残差: {rel_res:.4e}")
    print(f"自由度: {len(free_dofs)}, 非零元: {K.nnz}")
    return u, u_exact, coords, max_err, l2_rel_err

def plot_poisson_solution(u, coords, nx, ny, title):
    """绘制三维曲面图（需要 matplotlib）"""
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = coords[:,0].reshape(ny+1, nx+1)
        y = coords[:,1].reshape(ny+1, nx+1)
        u_reshaped = u.reshape(ny+1, nx+1)
        surf = ax.plot_surface(x, y, u_reshaped, cmap='viridis')
        ax.set_title(title)
        plt.show()
    except:
        print("无法显示图形，请检查 matplotlib 安装。")

# ==================== 统一接口：求解缩减方程 ====================

def solve_equilibrium(K_FF, rhs, method='ldlt', **options):
    """
    统一的求解接口，用于替代 2.3 中的简单求解
    method: 'ldlt' (稠密), 'sparse' (scipy.sparse.linalg.spsolve)
    """
    if method == 'ldlt':
        L, D = ldlt_factor(K_FF)
        d_F = ldlt_solve(L, D, rhs)
        return d_F
    elif method == 'sparse':
        # 假设 K_FF 是 numpy 数组，转为稀疏矩阵
        from scipy.sparse import csc_matrix
        K_sp = csc_matrix(K_FF)
        d_F = spsolve(K_sp, rhs)
        return d_F
    else:
        raise ValueError(f"不支持的求解方法: {method}")

# ==================== 主程序 ====================

def main_24():
    """运行 2.4 作业的所有算例"""
    print("="*60)
    print("2.4 平衡方程组求解器 - 完整测试")
    print("="*60)

    # ------- 复用 2.3 算例1：一维杆 -------
    print("\n" + "="*60)
    print("2.3 算例1：一维两单元杆 (LDL^T 求解)")
    print("="*60)
    data1 = {
        "nsd":1,"ndof":1,"nnp":3,"nel":2,"nen":2,
        "E":[100,200],"CArea":[1,1],
        "x":[0,1,2],"y":[0,0,0],
        "IEN":[[1,2],[2,3]],
        "fixed_dof":[1],"fixed_value":[0],
        "force_dof":[3],"force_value":[10]
    }
    nsd, ndof, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, force_dof, force_val = read_input_from_dict(data1)
    LM = generate_LM(IEN, ndof)
    K_full = assemble_global_stiffness(nnp, ndof, nel, nen, LM, IEN, E, A, x, y)
    K_FF, rhs, free_dof = reduce_system(K_full, fixed_dof, fixed_val, force_dof, force_val)
    print("缩减矩阵 K_FF =", K_FF)
    print("右端 rhs =", rhs)
    # 使用 LDLT 求解
    d_F = solve_equilibrium(K_FF, rhs, method='ldlt')
    d = np.zeros(nnp*ndof)
    d[fixed_dof] = fixed_val
    d[free_dof] = d_F
    print("位移 d =", d)
    # 后处理
    results = postprocess_1D(nel, IEN, E, A, x, d, ndof)
    for r in results:
        print(f"单元 {r['element']}: 应力 = {r['stress']:.8f}, 轴力 = {r['axial_force']:.8f}")
    # 反力
    R_all = K_full @ d
    print(f"约束反力 (自由度 {fixed_dof+1}): {R_all[fixed_dof]}")

    # ------- 2.3 算例2：二维桁架 -------
    print("\n" + "="*60)
    print("2.3 算例2：二维两杆桁架 (LDL^T 求解)")
    print("="*60)
    data2 = {
        "nsd":2,"ndof":2,"nnp":3,"nel":2,"nen":2,
        "E":[1,1],"CArea":[1,1],
        "x":[1,0,1],"y":[0,0,1],
        "IEN":[[1,3],[2,3]],
        "fixed_dof":[1,2,3,4],"fixed_value":[0,0,0,0],
        "force_dof":[5,6],"force_value":[10,0]
    }
    nsd, ndof, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, force_dof, force_val = read_input_from_dict(data2)
    LM = generate_LM(IEN, ndof)
    K_full = assemble_global_stiffness(nnp, ndof, nel, nen, LM, IEN, E, A, x, y)
    K_FF, rhs, free_dof = reduce_system(K_full, fixed_dof, fixed_val, force_dof, force_val)
    d_F = solve_equilibrium(K_FF, rhs, method='ldlt')
    d = np.zeros(nnp*ndof)
    d[fixed_dof] = fixed_val
    d[free_dof] = d_F
    print("节点位移 (u3,v3) =", d[4], d[5])
    results = postprocess_2D(nel, IEN, E, A, x, y, d, ndof)
    for r in results:
        print(f"单元 {r['element']}: 应力 = {r['stress']:.8f}, 轴力 = {r['axial_force']:.8f}")

    # ------- 算例2：病态矩阵 -------
    ill_conditioned_test()

    # ------- 算例4：Poisson 方程 (小规模对比) -------
    print("\n" + "="*60)
    print("Poisson 方程算例 (小规模对比 LDLT 与稀疏求解)")
    print("="*60)
    # 小网格 (n=10) 对比两种方法
    nx = 10; ny = 10
    # 用 LDLT
    u_ldlt, _, _, err_ldlt, l2_ldlt = solve_poisson(nx, ny, method='ldlt')
    # 用 scipy.sparse
    u_sparse, _, _, err_sparse, l2_sparse = solve_poisson(nx, ny, method='sparse')
    print("\n对比：")
    print(f"LDLT: 最大误差 = {err_ldlt:.4e}, L2相对误差 = {l2_ldlt:.4e}")
    print(f"Sparse: 最大误差 = {err_sparse:.4e}, L2相对误差 = {l2_sparse:.4e}")

    # ------- 大规模 Poisson (仅稀疏求解) -------
    print("\n" + "="*60)
    print("大规模 Poisson 算例 (nx=ny=50, 100, 200)")
    print("="*60)
    for n in [50, 100, 200]:
        solve_poisson(n, n, method='sparse')

    print("\n所有测试完成。")

if __name__ == "__main__":
    main_24()