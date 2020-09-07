# 作业解答

## 规划模型


- 请使用Python `scipy`库 的`optimize.linprog`方法，求解以下线性规划问题,并通过图解法验证。

$$
\begin{array}{l}
&{\max z= 4x_{1}+ 3x_{2}} \\
&\text { s.t. }{\quad\left\{\begin{array}{l}
{2x_{1}+ x_{2} \leq 10} \\ 
{x_{1}+ x_{2} \leq 8} \\ 
{x_{1}, x_{2} \geq 0}
\end{array}\right.}\end{array}
$$

#导入相关库
import numpy as np
from scipy import optimize as op

#定义决策变量范围
x1=(0,None)
x2=(0,None)

#定义目标函数系数(请注意这里是求最大值，而linprog默认求最小值，因此我们需要加一个符号)
c=np.array([-4,-3]) 

#定义约束条件系数
A_ub=np.array([[2,1],[1,1]])
B_ub=np.array([10,8])

#求解
res=op.linprog(c,A_ub,B_ub,bounds=(x1,x2))
res

- 请使用Python `scipy`库 的`optimize.minimize`方法，求解以下非线性规划问题


$$
\begin{array}{l}
&{\min z= x_{1}^2 + x_{2}^2 +x_{3}^2} \\
&\text { s.t. }{\quad\left\{\begin{array}{l}
{x_1+x_2 + x_3\geq9 } \\ 
{ x_{1}, x_{2},x_3 \geq 0}
\end{array}\right.}\end{array}
$$

import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return (x[0] ** 2 + x[1]**2 + x[2]**2)

# 定义约束条件
def constraint1(x):
    return (x[0]  + x[1] + x[2]  - 9)  # 不等约束1

# 汇总约束条件
con1 = {'type': 'ineq', 'fun': constraint1}
cons = ([con1])  

# 决策变量的符号约束
b = (0.0, None) #即决策变量的取值范围为大于等于0
bnds = (b, b ,b) 

#定义初始值
x0=np.array([0, 0, 0]) 

# 求解
solution = minimize(objective, x0, method='SLSQP', \
                    bounds=bnds, constraints=cons)
                    
x = solution.x

# 打印结果
print('目标值: ' + str(objective(x)))
print('最优解为')
print('x1 = ' + str(round(x[0],2)))
print('x2 = ' + str(round(x[1],2)))
print('x3 = ' + str(round(x[2],2)))
solution

- 某农场 I,II,III 等耕地的面积分别为 $100 hm^2$、$300 hm^2$ 和 $200 hm^2$，计划种植水稻、大豆和玉米，要求三种作物的最低收获量分别为$190000kg$、$130000kg$和$350000kg$。I,II,III 等耕地种植三种作物的单产如下表所示。
若三种作物的售价分别为水稻1.20元/kg，大豆1.50元/kg，玉米0.80元/kg。那么，
	- 如何制订种植计划才能使总产量最大？
	- 如何制订种植计划才能使总产值最大？
**要求：写出规划问题的标准型，并合理采用本课程学到的知识，进行求解。**


|         | I等耕地 | II等耕地| III等耕地 |
| :--------:| :--------: | :--------: | :--------: |
| 水稻     | 11000     | 9500     |  9000|
| 大豆     | 8000     | 6800     |  6000|
| 玉米     | 14000     | 12000     |  10000|

#导入相关库
import numpy as np
from scipy import optimize as op

#定义决策变量范围
x11=(0,None)
x12=(0,None)
x13=(0,None)

x21=(0,None)
x22=(0,None)
x23=(0,None)

x31=(0,None)
x32=(0,None)
x33=(0,None)
bounds=(x11,x12,x13,x21,x22,x23,x31,x32,x33)

#定义目标函数系数(请注意这里是求最大值，而linprog默认求最小值，因此我们需要加一个负号)
c=np.array([-11*1.2,-8*1.5,-14*0.8,-9.5*1.2,-6.9*1.5,-12*0.8,-9*1.2,-6*1.5,-10*0.8]) 

#定义不等式约束条件系数
A = - np.array([[11,0,0,9.5,0,0,9,0,0],[0,8,0,0,6.9,0,0,6,0],[0,0,14,0,0,12,0,0,10]])
b = - np.array([190,130,350])

#定义等式约束条件系数
A_eq = np.array([[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1]])
b_eq = np.array([100,300,200])


#求解
res=op.linprog(c,A,b,A_eq,b_eq,bounds=bounds)
res