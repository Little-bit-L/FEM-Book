import numpy as np
import json
import sys

# ===================== 前处理 =====================
def read_input_from_dict(data):
    """从字典读取模型数据（可直接从JSON加载）"""
    nsd = data['nsd']               # 空间维度 1 或 2
    ndof = data['ndof']             # 每个节点自由度数
    nnp = data['nnp']               # 节点总数
    nel = data['nel']               # 单元总数
    nen = data['nen']               # 每个单元节点数 (2)
    E = np.array(data['E'])         # 弹性模量数组 (nel,)
    A = np.array(data['CArea'])     # 截面积数组 (nel,)
    x = np.array(data['x'])         # 节点 x 坐标 (nnp,)
    y = np.array(data['y']) if nsd == 2 else np.zeros(nnp)
    IEN = np.array(data['IEN']) - 1 # 转换为0基索引 (nel, nen)
    fixed_dof = np.array(data['fixed_dof']) - 1   # 固定自由度编号(从1开始转0基)
    fixed_val = np.array(data['fixed_value'])
    force_dof = np.array(data['force_dof']) - 1
    force_val = np.array(data['force_value'])
    return nsd, ndof, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, force_dof, force_val

# ===================== 对号矩阵生成 =====================
def generate_LM(IEN, ndof):
    """生成对号矩阵 LM: (nel, nen*ndof) 每个单元的自由度全局编号"""
    nel, nen = IEN.shape
    LM = np.zeros((nel, nen*ndof), dtype=int)
    for e in range(nel):
        for a in range(nen):
            node = IEN[e, a]
            for i in range(ndof):
                LM[e, a*ndof + i] = node * ndof + i
    return LM

# ===================== 单元刚度矩阵 =====================
def element_stiffness_1D(E, A, L):
    """一维杆单元刚度矩阵 (2x2)"""
    return (E * A / L) * np.array([[1, -1], [-1, 1]])

def element_stiffness_2D(E, A, L, cos, sin):
    """二维桁架单元刚度矩阵 (4x4)"""
    c, s = cos, sin
    k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
    T = np.array([[c, s, 0, 0],
                  [0, 0, c, s]])
    return T.T @ k_local @ T

# ===================== 总体刚度矩阵组装 =====================
def assemble_global_stiffness(nnp, ndof, nel, nen, LM, IEN, E, A, x, y):
    """直接组装总体刚度矩阵"""
    N = nnp * ndof
    K = np.zeros((N, N))
    for e in range(nel):
        node1 = IEN[e, 0]
        node2 = IEN[e, 1]
        L = np.sqrt((x[node2] - x[node1])**2 + (y[node2] - y[node1])**2)
        if ndof == 1:
            Ke = element_stiffness_1D(E[e], A[e], L)
        else:  # ndof == 2
            dx = x[node2] - x[node1]
            dy = y[node2] - y[node1]
            cos = dx / L
            sin = dy / L
            Ke = element_stiffness_2D(E[e], A[e], L, cos, sin)
        # 组装
        dof_list = LM[e]
        for i in range(len(dof_list)):
            for j in range(len(dof_list)):
                K[dof_list[i], dof_list[j]] += Ke[i, j]
    return K

# ===================== 方程求解（缩减法） =====================
def solve_reduction(K, fixed_dof, fixed_val, force_dof, force_val):
    """缩减法求解位移和反力"""
    N = K.shape[0]
    all_dof = np.arange(N)
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    K_FF = K[np.ix_(free_dof, free_dof)]
    K_FE = K[np.ix_(free_dof, fixed_dof)]
    F_F = np.zeros(N)
    F_F[force_dof] = force_val
    F_F = F_F[free_dof]

    d_F = np.linalg.solve(K_FF, F_F - K_FE @ fixed_val)
    d = np.zeros(N)
    d[fixed_dof] = fixed_val
    d[free_dof] = d_F

    R = K @ d
    R_fixed = R[fixed_dof]
    return d, R_fixed

# ===================== 后处理 =====================
def postprocess_1D(nel, IEN, E, A, x, d, ndof):
    """一维后处理：单元应力、轴力"""
    results = []
    for e in range(nel):
        node1 = IEN[e, 0]
        node2 = IEN[e, 1]
        L = x[node2] - x[node1]   # 一维长度就是坐标差
        u1 = d[node1 * ndof]
        u2 = d[node2 * ndof]
        de = np.array([u1, u2])
        sigma = E[e] / L * np.array([-1, 1]) @ de
        force = sigma * A[e]
        results.append({
            'element': e+1,
            'length': L,
            'stress': sigma,
            'axial_force': force
        })
    return results

def postprocess_2D(nel, IEN, E, A, x, y, d, ndof):
    """二维后处理：单元应力、轴力、方向余弦"""
    results = []
    for e in range(nel):
        node1 = IEN[e, 0]
        node2 = IEN[e, 1]
        dx = x[node2] - x[node1]
        dy = y[node2] - y[node1]
        L = np.sqrt(dx**2 + dy**2)
        cos = dx / L
        sin = dy / L
        dof_idx = [node1*ndof, node1*ndof+1, node2*ndof, node2*ndof+1]
        de = d[dof_idx]
        c, s = cos, sin
        sigma = E[e] / L * np.array([-c, -s, c, s]) @ de
        force = sigma * A[e]
        results.append({
            'element': e+1,
            'length': L,
            'cos': cos,
            'sin': sin,
            'stress': sigma,
            'axial_force': force
        })
    return results

def check_singular(K):
    """检查矩阵奇异性（特征值接近0）"""
    eigvals = np.linalg.eigvals(K)
    return np.any(np.abs(eigvals) < 1e-10)

# ===================== 主程序 =====================
def main(input_data):
    # 前处理
    nsd, ndof, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, force_dof, force_val = read_input_from_dict(input_data)

    # 生成对号矩阵
    LM = generate_LM(IEN, ndof)
    print("对号矩阵 LM (每行对应单元的自由度全局编号):")
    print(LM + 1)  # 转回1基显示

    # 组装总体刚度矩阵
    K = assemble_global_stiffness(nnp, ndof, nel, nen, LM, IEN, E, A, x, y)
    print("\n总体刚度矩阵 K:")
    print(K)
    print(f"是否对称: {np.allclose(K, K.T)}")
    print(f"施加边界条件前奇异? {check_singular(K)}")

    # 求解
    d, R_fixed = solve_reduction(K, fixed_dof, fixed_val, force_dof, force_val)
    print("\n节点位移 (所有自由度):")
    for i in range(nnp):
        if ndof == 1:
            print(f"节点 {i+1}: u = {d[i]:.8f}")
        else:
            print(f"节点 {i+1}: u = {d[2*i]:.8f}, v = {d[2*i+1]:.8f}")

    print("\n约束反力 (固定自由度):")
    for idx, dof in enumerate(fixed_dof):
        dof_1based = dof + 1
        print(f"自由度 {dof_1based}: 反力 = {R_fixed[idx]:.8f}")

    # 后处理应力
    if ndof == 1:
        results = postprocess_1D(nel, IEN, E, A, x, d, ndof)
    else:
        results = postprocess_2D(nel, IEN, E, A, x, y, d, ndof)
    print("\n单元计算结果:")
    for res in results:
        print(f"单元 {res['element']}: 长度 = {res['length']:.6f}, 应力 = {res['stress']:.8f}, 轴力 = {res['axial_force']:.8f}")
        if ndof == 2:
            print(f"         方向余弦: cos={res['cos']:.6f}, sin={res['sin']:.6f}")


# ===================== 入口（支持命令行参数，无参数时演示两个算例） =====================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 从命令行读取 JSON 文件
        json_file = sys.argv[1]
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功读取输入文件: {json_file}\n")
            main(data)
        except FileNotFoundError:
            print(f"错误：文件 '{json_file}' 未找到。")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"错误：文件 '{json_file}' 不是有效的 JSON 格式。")
            sys.exit(1)
    else:
        # 无参数时，依次演示算例1和算例2
        print("未指定输入文件，将依次演示算例1（一维两单元杆）和算例2（二维两杆桁架）。\n")

        # 算例1 数据
        example1_data = {
            "Title": "一维两单元杆结构",
            "nsd": 1, "ndof": 1, "nnp": 3, "nel": 2, "nen": 2,
            "E": [100, 200],
            "CArea": [1, 1],
            "x": [0, 1, 2],
            "y": [0, 0, 0],
            "IEN": [[1, 2], [2, 3]],
            "fixed_dof": [1],
            "fixed_value": [0],
            "force_dof": [3],
            "force_value": [10]
        }

        # 算例2 数据
        example2_data = {
            "Title": "二维两杆桁架结构",
            "nsd": 2, "ndof": 2, "nnp": 3, "nel": 2, "nen": 2,
            "E": [1.0, 1.0],
            "CArea": [1.0, 1.0],
            "x": [1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0],
            "IEN": [[1, 3], [2, 3]],
            "fixed_dof": [1, 2, 3, 4],
            "fixed_value": [0.0, 0.0, 0.0, 0.0],
            "force_dof": [5, 6],
            "force_value": [10.0, 0.0]
        }

        # 运行算例1
        print("=" * 60)
        print("算例1：一维两单元杆结构")
        print("=" * 60)
        main(example1_data)

        # 运行算例2
        print("\n" + "=" * 60)
        print("算例2：二维两杆桁架结构")
        print("=" * 60)
        main(example2_data)