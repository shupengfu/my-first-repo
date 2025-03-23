import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

#定义真解函数
def true_solution(x):
    return np.exp(-x**2) * (1-x**2)

#定义右端项f(x)
def f(x):
    u = true_solution(x)
    u_xx = np.gradient(np.gradient(u,x,edge_order=2),x,edge_order=2)
    return -u_xx + u

#定义有限差分格式的线性方程组，并求解
def finite_difference(h):
    N = int(2 / h)-1  #定义内部节点数
    x = np.linspace(-1,1,N+2)   #定义x的坐标，包括边界点
    x_internal = x[1: -1]   #索引从第一个元素到倒数第二个元素
    #定义系数矩阵
    D = np.zeros((3,N))
    D[0, :] = -1  # 上对角线
    D[1, :] = 2 + h**2  # 主对角线
    D[2, :] = -1  # 下对角线
    A = diags(D, [-1,0,1],(N,N))
    #定义方程组右端向量
    F = h**2 * f(x_internal)
    #数值求解u
    U_internal = spsolve(A.tocsc(),F)
    U = np.zeros(N + 2)
    U[1:-1] = U_internal
    return x,U

#设置误差以及各种参数列表
h_list = [1/10, 1/20, 1/40, 1/80]
error_l2 = []
error_lif = []
x_list = []
u_num_list = []
u_true_list = []

for h in h_list:
    x,u_num = finite_difference(h)
    u_true = true_solution(x)
    error = u_num - u_true
    error_l2.append(np.linalg.norm(error, ord=2))
    error_lif.append(np.max(np.abs(error)))
    x_list.append(x)
    u_num_list.append(u_num)
    u_true_list.append(u_true)

#计算收敛阶数,np.polyfit(x,y,1)会将两个值拟合p=1的线性函数
# poly是一个二元数组(斜率，截距)，poly[0]返回第0个元素————斜率
def convergent_order(errors,h_list,p):
    errors = np.array(errors)
    h_list = np.array(h_list)
    log_errors = np.log(errors)
    log_h = np.log(h_list)
    poly = np.polyfit(log_h,log_errors,1)
    order = poly[0]
    return order

#返回收敛阶
order_l2 = convergent_order(error_l2,h_list,2)
order_lif = convergent_order(error_lif,h_list,np.inf)

#绘制图像
plt.figure(figsize=(10,10))
#绘制真解函数图像
plt.subplot(2,2,1)
x_ture1 = np.linspace(-1,1,500)
y_ture1 = true_solution(x_ture1)
plt.plot(x_ture1,y_ture1,label='True Solution')
plt.title('True Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()

#绘制数值解函数图像
plt.subplot(2,2,2)
for i, h in enumerate(h_list):
    plt.plot(x_list[i], u_num_list[i], label=f'h={h}')
plt.title('Numerical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()

#绘制误差曲线图像
plt.subplot(2, 2, 3)
for i, h in enumerate(h_list):
    plt.plot(x_list[i], np.abs(u_num_list[i] - u_true_list[i]), label=f'h={h}')
plt.title('Error Curves')
plt.xlabel('x')
plt.ylabel('|u_num - u_true|')
plt.legend()

plt.tight_layout()
plt.show()

#输出误差和收敛阶数
print("l^2误差：",error_l2)
print("\nl^inf误差：",error_lif)
print("\nl^2收敛阶数：",order_l2)
print("\nl^inf收敛阶数：",order_lif)