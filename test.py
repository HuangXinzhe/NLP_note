# 

import numpy as np

def Viterbi(A, B, PI, V, Q, obs):

    N = len(Q)  # 状态类型数量
    T = len(obs)  # 观察状态的长度
    delta = np.array([[0] * N] * T, dtype=np.float64)  # delta[i][j]表示i时刻状态为Q[j]的所有路径中，概率最大的路径的概率值
    phi = np.array([[0] * N] * T, dtype=np.int64)  # phi[i][j]表示i时刻状态为Q[j]的所有路径中，概率最大的路径中，i-1时刻的状态
    # 初始化
    for i in range(N):
        delta[0, i] = PI[i]*B[i][V.index(obs[0])]
        phi[0, i] = 0

    # 递归计算
    for i in range(1, T):
        for j in range(N):
            tmp = [delta[i-1, k]*A[k][j] for k in range(N)]  # 前一个时刻状态到当前状态概率
            delta[i,j] = max(tmp) * B[j][V.index(obs[i])]  # 当前状态的概率
            phi[i,j] = tmp.index(max(tmp))  # 当前状态的前一个状态

    # 最终的概率及节点
    P = max(delta[T-1, :])  # 最大的概率
    I = int(np.argmax(delta[T-1, :]))  # 最大概率对应的节点

    # 最优路径path
    path = [I]
    for i in reversed(range(1, T)):
        end = path[-1]
        path.append(phi[i, end])

    hidden_states = [Q[i] for i in reversed(path)]

    return P, hidden_states


def main():

    # 状态集合
    Q = ('欢乐谷', '迪士尼', '外滩')
    # 观测集合
    V = ['购物', '不购物']
    # 转移概率: Q -> Q
    A = [[0.8, 0.05, 0.15],
         [0.2, 0.6, 0.2],
         [0.2, 0.3, 0.5]
        ]

    # 发射概率, Q -> V
    B = [[0.1, 0.9],
         [0.8, 0.2],
         [0.3, 0.7]
         ]

    # 初始概率
    PI = [1/3, 1/3, 1/3]

    # 观测序列
    obs = ['不购物', '购物', '购物']

    P, hidden_states = Viterbi(A,B,PI,V,Q,obs)
    print('最大的概率为: %.5f.'%P)
    print('隐藏序列为：%s.'%hidden_states)

if __name__ == '__main__':
    main()