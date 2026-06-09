import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元的单元刚度矩阵（全局坐标系）

    参数:
        x1, x2: 两个节点的坐标，格式为 [x, y, z] (list or array)
        E: 弹性模量 (Pa)
        A: 截面积 (m^2)

    返回:
        L: 单元长度 (m)
        direction_cosines: 方向余弦 [cx, cy, cz]
        Ke: 6x6 单元刚度矩阵 (N/m)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    delta = x2 - x1
    L = np.linalg.norm(delta)

    if L < 1e-12:
        raise ValueError("错误：两个节点重合，单元退化！")

    cx, cy, cz = delta / L   # 方向余弦

    # 计算刚度矩阵
    k = E * A / L
    c = np.array([-cx, -cy, -cz, cx, cy, cz])  # 应变-位移矩阵的系数 (1/L)*c
    Ke = k * np.outer(c, c)

    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元的应变、应力和轴力

    参数:
        x1, x2: 节点坐标
        E, A: 材料属性
        de: 单元节点位移列阵，形状为 (6,) 或 (6,1)，顺序 [u1, v1, w1, u2, v2, w2]

    返回:
        epsilon: 轴向应变 (无量纲)
        sigma: 轴向应力 (Pa)
        N: 轴力 (N)，拉为正
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    de = np.asarray(de, dtype=float).flatten()
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("错误：两个节点重合，无法计算应力！")

    cx, cy, cz = delta / L
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L   # 应变-位移矩阵 (1x6)
    epsilon = np.dot(B, de)
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N


# ===================== 验证算例 =====================
def validate_example1():
    print("=" * 60)
    print("算例1：沿x轴的一维杆单元")
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9      # 200 GPa
    A = 1.0e-4     # m^2
    de = [0, 0, 0, 1.0e-3, 0, 0]   # 节点位移

    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    print(f"单元长度 L = {L:.6f} m (期望 2.0 m)")
    print(f"方向余弦: ({cx:.0f}, {cy:.0f}, {cz:.0f}) 期望 (1, 0, 0)")
    print("刚度矩阵 Ke (6x6):")
    print(np.array2string(Ke, precision=2, suppress_small=True))

    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    print(f"应变 epsilon = {epsilon:.6e} (期望 5.0e-4)")
    print(f"应力 sigma = {sigma/1e6:.2f} MPa (期望 100 MPa)")
    print(f"轴力 N = {N:.2f} N (期望 10000 N)")
    print()


def validate_example2():
    print("=" * 60)
    print("算例2：空间任意方向杆单元")
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9
    A = 2.0e-4
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    print(f"单元长度 L = {L:.6f} m (期望 3.0 m)")
    print(f"方向余弦: ({cx:.4f}, {cy:.4f}, {cz:.4f}) 期望 (1/3≈0.3333, 2/3≈0.6667, 2/3≈0.6667)")

    # 检查对称性
    is_symmetric = np.allclose(Ke, Ke.T)
    print(f"刚度矩阵是否对称: {is_symmetric}")

    # 检查刚体平移（所有节点位移相同）
    de_rigid = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    F_rigid = Ke @ de_rigid
    print(f"刚体平移产生的节点力 (应接近零): {F_rigid}")

    # 特征值分析
    eigvals = np.linalg.eigvalsh(Ke)
    eigvals[np.abs(eigvals) < 1e-10] = 0.0
    print(f"刚度矩阵特征值: {eigvals}")
    print("解释：单个自由杆单元只有轴向刚度，因此只有一个正特征值，其余五个特征值为零，矩阵奇异。")

    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    print(f"应变 epsilon = {epsilon:.6e} (期望 1.0e-3)")
    print(f"应力 sigma = {sigma/1e6:.2f} MPa (期望 210 MPa)")
    print(f"轴力 N = {N:.2f} N (期望 42000 N)")
    print()


# ===================== 任务4：刚度矩阵物理意义验证 =====================
def verify_physical_meaning():
    print("=" * 60)
    print("任务4：刚度矩阵物理意义验证")
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9
    A = 1e-4
    L, _, Ke = truss3d_element_stiffness(x1, x2, E, A)

    # 选择自由度 j = 1 (即第二个自由度，对应节点1的y方向位移)
    j = 1   # 0-index 对应 u1? 我们按顺序：自由度 0:u1, 1:v1, 2:w1, 3:u2, 4:v2, 5:w2
    de = np.zeros(6)
    de[j] = 1.0   # 令第 j 个自由度的位移为1，其余为0
    F = Ke @ de
    print(f"令自由度 {j} (节点1的v方向) 位移为1，其余为0时，节点力列阵:")
    print(F)
    print("该节点力列阵恰好等于刚度矩阵的第", j, "列 (索引从0开始)。")
    print("物理意义：k_ij 表示第 j 个自由度产生单位位移时，在第 i 个自由度上需要施加的节点力。")
    print()


# ===================== 附加题：稳定空间桁架（四面体） =====================
def example_truss_assembly():
    print("=" * 60)
    print("附加题：空间四面体桁架组装与求解")
    # 节点坐标
    nodes = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0]),
        3: np.array([0.0, 2.0, 0.0]),
        4: np.array([0.0, 0.0, 2.0])
    }
    # 杆件：四面体的6条棱
    elements = [(1,2), (1,3), (1,4), (2,3), (3,4), (4,2)]
    E = 200e9
    A = 1e-4

    dof_per_node = 3
    n_nodes = len(nodes)
    n_dof = n_nodes * dof_per_node
    K_global = np.zeros((n_dof, n_dof))

    # 组装全局刚度矩阵
    for (n1, n2) in elements:
        x1 = nodes[n1]
        x2 = nodes[n2]
        _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)
        dofs = [dof_per_node*(n1-1) + i for i in range(dof_per_node)] + \
               [dof_per_node*(n2-1) + i for i in range(dof_per_node)]
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K_global[gi, gj] += Ke[i, j]

    # 边界条件：节点1、2、3完全固定（自由度0~8）
    fixed_dofs = list(range(0, 9))   # 节点1: 0,1,2 ; 节点2: 3,4,5 ; 节点3: 6,7,8
    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]

    # 载荷：节点4 (第4个节点) 的 z 方向受力 -10000 N
    F_global = np.zeros(n_dof)
    node4_dof_z = dof_per_node*(4-1) + 2   # z 方向索引2
    F_global[node4_dof_z] = -10000.0

    # 求解
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F_global[free_dofs]
    u_f = np.linalg.solve(K_ff, F_f)

    # 完整位移向量
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f

    # 输出节点位移
    print("节点位移 (m):")
    for node in range(1, n_nodes+1):
        idx = dof_per_node*(node-1)
        disp = u_global[idx:idx+3]
        print(f"  节点{node}: ({disp[0]:.6e}, {disp[1]:.6e}, {disp[2]:.6e})")

    # 计算各杆的应力和轴力
    print("\n各杆内力:")
    for (n1, n2) in elements:
        x1 = nodes[n1]
        x2 = nodes[n2]
        idx1 = dof_per_node*(n1-1)
        idx2 = dof_per_node*(n2-1)
        de = np.concatenate([u_global[idx1:idx1+3], u_global[idx2:idx2+3]])
        eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
        print(f"  杆 {n1}-{n2}: 应变={eps:.6e}, 应力={sig/1e6:.2f} MPa, 轴力={N:.2f} N")
    print()


if __name__ == "__main__":
    validate_example1()
    validate_example2()
    verify_physical_meaning()
    example_truss_assembly()