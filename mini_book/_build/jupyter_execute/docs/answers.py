# 作业解答


## 矩阵和线性代数基础

已知

$$
A=\left[\begin{array}{lll}
1 & 2 & 3 \\
2 & 3 & 4 \\
4 & 5 & 6
\end{array}\right], B=\left[\begin{array}{lll}
1 & 3 & 1 \\
2 & 1 & 1 \\
1 & 1 & 4
\end{array}\right]
$$


那么

$$
A + B = \left[\begin{array}{lll}
2 & 5 & 4 \\
4 & 4 & 5 \\
5 & 6 & 10
\end{array}\right]
$$

$$
A - B = \left[\begin{array}{lll}0 & -1 & 2 \\0 & 2 & 3 \\3 & 4 & 2\end{array}\right]
$$


$$
3A - 2B = \left[\begin{array}{lll}1 & 0 & 7 \\2 & 7 & 10 \\10 & 13 & 10\end{array}\right]
$$

$$
AB = \left[\begin{array}{lll}8 & 8 & 15 \\12 & 13 & 21 \\20 & 23 & 33\end{array}\right]
$$


$$
(A+2B)A = \left[\begin{array}{lll}39 & 55 & 71 \\40 & 57 & 74 \\76 & 103 & 130\end{array}\right]
$$














## 规划模型


``` {admonition} 作业1
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

```

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

``` {admonition} 作业2
- 请使用Python `scipy`库 的`optimize.minimize`方法，求解以下非线性规划问题


$$
\begin{array}{l}
&{\min z= x_{1}^2 + x_{2}^2 +x_{3}^2} \\
&\text { s.t. }{\quad\left\{\begin{array}{l}
{x_1+x_2 + x_3\geq9 } \\ 
{ x_{1}, x_{2},x_3 \geq 0}
\end{array}\right.}\end{array}
$$

```

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

``` {admonition} 作业3
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

```

设第$i$个农场种第$j$种作物的量为$x_{ij}$,三个农场，三种作物，因此决策变量一共有九个

$$
\left(
\begin{matrix}
x_{11}& x_{12} & x_{13}\\
x_{21}& x_{22} & x_{23}\\
x_{31}& x_{32} & x_{33}\\
\end{matrix}
\right)
$$


追求产量最大时，目标函数为

$$
\max 1000\left(11x_{11} + 8x_{12}+14 x_{13} + 9.5x_{21}+6.9x_{22}+12x_{23}+9x_{31}+6x_{32}+10x_{33} \right)
$$


约束条件为最低产量约束和种植面积约束

$$
s.t.
\left\{
\begin{aligned}
11 x_{11} + 9.5x_{21} + 9x_{31} \geq 190\\
8 x_{12} + 6.9x_{22} + 6x_{32} \geq 130\\
14 x_{13} + 12x_{23} + 10x_{33} \geq 350\\
x_{11} + x_{12} + x_{13} = 100 \\
x_{21} + x_{22} + x_{23} = 300 \\
x_{31} + x_{32} + x_{33} = 200 \\
\end{aligned}\right.
$$


除此之外，还有正值约束

$$
x_{ij}\geq 0
$$

编程求解如下

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

## 预测模型



1. 用如下代码生成一个包含了随机变动的正弦函数曲线，请你使用多项式拟合方法，研究用二次，三次，以及更高次函数拟合的情况，给你你认为的最好的拟合方法。

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
x = [0.1 * i for i in range(100)]
y = [np.sin(t) + np.random.random() for t in x]
plt.scatter(x,y)

### 2次

from scipy.optimize import curve_fit  # 导入非线性拟合函数curve_fit

# 定义需要拟合的函数形式，这里使用二次函数的一般式 y = ax^2 + bx + c
def f2(x, a, b, c):
    return a * x**2 + b*x + c


plt.scatter(x, y)  # 绘制散点图
popt, pcov = curve_fit(f2, x, y)    # 执行非线性拟合
# popt数组中，三个值分别是待求参数a,b,c
y1 = [f2(i, popt[0], popt[1], popt[2]) for i in x]   # 计算得到拟合曲线上的一系列点
plt.plot(x, y1, 'r')   # 绘制拟合曲线

### 3次

from scipy.optimize import curve_fit  # 导入非线性拟合函数curve_fit

# 定义需要拟合的函数形式，这里使用二次函数的一般式 y = ax^2 + bx + c
def f2(x, a, b, c,d):
    return a * x**3 + b*x**2 + c*x +d


plt.scatter(x, y)  # 绘制散点图
popt, pcov = curve_fit(f2, x, y)    # 执行非线性拟合
# popt数组中，三个值分别是待求参数a,b,c
y1 = [f2(i, popt[0], popt[1], popt[2], popt[3]) for i in x]   # 计算得到拟合曲线上的一系列点
plt.plot(x, y1, 'r')   # 绘制拟合曲线

### 4次

from scipy.optimize import curve_fit  # 导入非线性拟合函数curve_fit

# 定义需要拟合的函数形式，这里使用二次函数的一般式 y = ax^2 + bx + c
def f2(x, a, b, c,d,e):
    return a * x**4 + b*x**3 + c*x**2 +d*x +e


plt.scatter(x, y)  # 绘制散点图
popt, pcov = curve_fit(f2, x, y)    # 执行非线性拟合
# popt数组中，三个值分别是待求参数a,b,c
y1 = [f2(i, popt[0], popt[1], popt[2], popt[3],popt[4]) for i in x]   # 计算得到拟合曲线上的一系列点
plt.plot(x, y1, 'r')   # 绘制拟合曲线

四次多项式已经比较能够描述数据的变化趋势。



2.给出下面的`interest rate`，`unemployment rate`和`stock index price`的数据，请你通过多元线性回归的方法，通过`interest rate`，`unemployment rate`来预测`stock index price`。给出你所得到的计算公式。你可以调用 `sklearn.linear_model`中的 `LinearRegression` 函数，关于该函数的使用方法，可以参考官方文档。
![Image Name](https://cdn.kesci.com/upload/image/qgbqq4833c.png?imageView2/0/w/960/h/960)

## 输入数据
interest_rate = [2.75,2.5,2.5,2.5,2.5,2.5,
                2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,
                1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,]
                
unemployment_rate = [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,
                    5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,
                    6.1,5.9,6.2,6.2,6.1]
                    
stock_index_price = [1464,1394,1357,1293,1256,1254,1234,1195,1159,
                    1167,1130,1075,1047,965,943,958,971,949,884,
                    866,876,822,704,719]

X = [interest_rate,unemployment_rate]
X = np.array(X).T
Y = stock_index_price

from sklearn.linear_model import LinearRegression  # 导入线性回归函数LinearRegression
lrModel = LinearRegression()     # 初始化回归模型
lrModel.fit(np.array(X),Y)  # 输入需要回归的数据

print('截距为：', lrModel.intercept_)  # 输出截距
print('系数为：', lrModel.coef_)    # 输出系数
score = lrModel.score(X,Y)
print('R2为：', score)    # 输出相关系数R2

$$
S = 345.54 I - 250 U + 1789
$$

我们来对比预测值和真实值

## 预测值
import matplotlib.pyplot as plt
Predicted_value = 345.54 * np.array(interest_rate) - 250 * np.array(unemployment_rate) + 1789
plt.plot(Predicted_value,label = 'Predicted_value')
plt.plot(Y,label = 'Observation')
plt.legend()

效果很不错，我们来看一下他们的平均相对误差。

np.mean((Predicted_value - Y)/Y)

只有0.2%的误差。



## 排队论模型


**问题1.** 某医院手术室根据病人来诊和完成手术时间的记录，经统计分析算出每小时病人平均到达率为$2.1人/h$，为泊松分布。每次手术时间$2.5人/h$，服从负指数分布。求:
- 病房中病人的平均数($L$)。
- 排队等待手术病人的平均数( $\left.L_{q}\right)$
- 病人在病房中平均逗留时间($W$)。
- 病人排队等待时间(期望值队 $\left.W_{q}\right)$

**问题2.** 到达某铁路售票处顾客分两类：一类买南方线路票，到达率为$\lambda_1$/小时，另一类买北方线路票，到达率为$\lambda_2$/小时，以上均服从泊松分布。该售票处设两个窗口，各窗口服务一名顾客时间均服从参数$\mu=10$的指数分布。试比较下列情况时顾客分别等待时间：
- 两个窗口分别售南方票和北方票；
- 每个窗口两种票均出售。（分别比较 $\lambda_1 = \lambda_2 = 2,4,6,8,10$时的情形）

### 问题1
该手术室为 $M / M /1/$系统

$$
\lambda=2.1 \text { 人/h, } 
$$

$$
\mu=2.5 \text { 人/h } 
$$

$$
\rho=\frac{\lambda}{\mu}=\frac{2.1}{2.5}=0.84
$$

(1)病房中病人的平均数: 

$$
L=\frac{\lambda}{\mu-\lambda}=\left(\frac{2.1}{2.5-2.1}\right)人=5.25 人
$$

(2)排队等待手术病人的平均数: 

$$
\quad L_{q} =\frac{\lambda^{2}}{\mu(\mu-\lambda)}=4.41人
$$

(3)病人在病房中平均逗留的时间: 

$$
\quad W=\frac{1}{\mu-\lambda}=\left(\frac{1}{2.5-2.1}\right) h=2.5 h
$$

(4)病人排队等待时间: 

$$
\quad W_{q}=W_{\rho}=\frac{\lambda}{\mu(\mu-\lambda)}=2.1 h
$$

### 问题2

如果是分售南方和北方票的话，就是两个$M/M/1$模型，$\mu = 10$，在这个模型中，计算排队时间的公式为

$$
W_q = \frac{\lambda}{\mu(\mu-\lambda)}
$$

如果是合并发售南方和北方票的话，就是一个$M/M/2$模型，在这个模型中，$c=2$，计算排队长度的公式为

$$
Wq = \frac{L_q}{\lambda}=\frac{(c \rho)^{c} \rho}{\lambda c !(1-\rho)^{2}} P_{0}
$$

其中，

$$
P_{0}=\left[\sum_{k=0}^{c-1} \frac{1}{k !}\left(\frac{\lambda}{\mu}\right)^{k}+\frac{1}{c !} \frac{1}{1-\rho}\left(\frac{\lambda}{\mu}\right)^{c}\right]^{-1}
$$

带入数据计算得到

$$
\begin{array}{|l|l|l|}
\hline \lambda_{1}=\lambda_{2} \text { 的值 } & \text { (a) 分售南方和北方票 } & \text { (b) 联合售票 } \\
\hline 2 & 0.025 & 0.004167 \\
4 & 0.0667 & 0.0190 \\
6 & 0.15 & 0.05625 \\
8 & 0.40 & 0.17777 \\
\hline
\end{array}
$$

计算代码如下

## 单队伍
import numpy as np
Lambda = 2
mu = 10
W_q = Lambda/(mu*(mu-Lambda))
W_q

## 多队伍
import numpy as np
Lambda = 4
c = 2
mu = 10
rho = Lambda/(c*mu)
S = 0
for k in range(c):
    S = S + 1/np.math.factorial(k)*(Lambda/mu)**k
P_0 = (S + 1/np.math.factorial(c)/(1-rho)*(Lambda/mu)**c)**(-1)
W_q = (c*rho)**c * rho/(Lambda *np.math.factorial(c)*(1-rho)**2 ) * P_0
W_q

## 微分方程模型

```{admonition} 微分方程模型作业
考虑种群竞争模型

$$
\left\{
\begin{aligned}
& \dfrac{\mathrm{d}x_1}{\mathrm{d}t}=r_{1} x_{1}\left(1-\frac{x_{1}}{N_{1}}-\sigma_{1} \frac{x_{2}}{N_{2}}\right)\\
& \dfrac{\mathrm{d}x_2}{\mathrm{d}t}=r_{2} x_{2}\left(1-\sigma_{2} \frac{x_{1}}{N_{1}}-\frac{x_{2}}{N_{2}}\right)
\end{aligned}\right.
$$

取$r_1 = 0.2, r_2 = 0.3, \sigma_1 = 1.2,\sigma_2 = 0.5,N_1 = 100,N_2 = 70, x_1(0) = 30,x_2(0) = 40$,使用本节课程学到的数值方法研究两个种群的发展模式。

```

import matplotlib.pyplot as plt
%matplotlib inline

r1 = 0.2
r2 = 0.3
N1 = 100
N2 = 70
sigma1 = 1.2
sigma2 = 0.5
x1_0 = 30
x2_0 = 40

deltaT = 0.01
TotTime = 40

timeStep = TotTime/deltaT

x1_list = []
x2_list = []
x1_list.append(x1_0)
x2_list.append(x2_0)

TimeList = [i*deltaT for i in range(int(timeStep))]

for time in TimeList:
    x1_list.append(x1_list[-1] +deltaT * (r1 *x1_list[-1])*(1 - x1_list[-1]/N1 - sigma1 * x2_list[-1]/N2))
    x2_list.append(x2_list[-1] +deltaT * (r2 *x2_list[-1])*(1 - sigma2* x1_list[-1]/N1 -  x2_list[-1]/N2))
plt.figure(figsize = (20,5))
plt.plot(TimeList,x1_list[:-1])
plt.plot(TimeList,x2_list[:-1])

## 马氏链作业


**问题1： ** 在英国，工党成员的第二代加入工党的概率为0.5，加入保守党的概率为0.4，加入自由党的概率为0.1。而保守党成员的第二代加入工党的概率为0.7，加入保守党的概率为0.2，加入自由党的概率为0.1。而自由党成员的第二代加入工党的概率为0.2，加入保守党的概率为0.4，加入自由党的概率为0.4。
也就是说，其转移概率矩阵为：

|      	|   	|     	| 下一代党派 	|     	|
|------	|---	|-----	|------	|-----	|
|      	|   	| 工党   	| 保守党    	| 自由党   	|
|      	| 工党 	| 0.5 	| 0.4  	| 0.1 	|
| 上一代党派 	| 保守党	| 0.7 	| 0.2  	| 0.1 	|
|      	| 自由党 	| 0.2 	| 0.4  	| 0.4 	|


- 求自由党成员的第三代加入工党的概率是多少？
- 在经过较长的时间后，各党成员的后代加入各党派的概率分布是否具有稳定性？




**问题2： ** 社会学的某些调查结果指出：儿童受教育的水平依赖于他们父母受教育的水平。调查过程是将人们划分为三类：$E$ 类，这类人具有初中或初中以下的文化程度；$S$类，这类人具有高中文化程度；$C$ 类，这类人受过高等教育。当父或母（指文化程度较高者）是这三类人中某一类型时，其子女将属于这三种类型中的任一种的概率由下面给出

|      	|   	|     	| 孩子 	|     	|
|------	|---	|-----	|------	|-----	|
|      	|   	| $E$   	| $S$    	| $C$   	|
|      	| $E$ 	| 0.7 	| 0.2  	| 0.1 	|
| 父母 	| $S$ 	| 0.4 	| 0.4  	| 0.2 	|
|      	| $C$ 	| 0.1 	| 0.2  	| 0.7 	|
问：
- 属于$S$ 类的人们中，其第三代将接受高等教育的概率是多少？
- 假设不同的调查结果表明，如果父母之一受过高等教育，那么他们的子女总
可以进入大学，修改上面的转移矩阵。
- 根据2的解，每一类型人的后代平均要经过多少代，最终都可以接受高
等教育？

### 第一题

本问题的转移概率矩阵为

$$
P=\left[\begin{array}{cccc}
{0.5} & {0.4} & {0.1}  \\ 
{0.7} & {0.2} & {0.1} \\ 
{0.2} & {0.4} & {0.4}.
\end{array}\right]
$$

自由党成员的第三代加入工党的概率

$$
p = 0.2 \times 0.5 (自-工-工) + 0.4 \times 0.7 (自-保-工) + 0.4 \times 0.2 (自-自-工) = 0.46
$$
接下来考察经过较长的时间后，各党成员的后代概率分布的稳定性。可以通过解析求解和计算机求解两个方法，先来看解析求解
假设各党成员的概率分布存在稳定解，且其稳定值为$p_1,p_2,p_3$，那么有如下条件成立

$$
\left\{\begin{array}{l}
{p_{1}=0.5 p_{1}+0.7 p_{2}+0.2 p_{3}} \\ 
{p_{2}=0.4 p_{1}+0.2 p_{2}+0.4 p_{3}} \\ 
{p_{3}=0.1 p_{1}+0.1 p_{2}+0.4 p_{3}} \end{array}\right.
$$

并且

$$
p_1+p_2 + p_3 = 1
$$

解得

$$
p_1 =\frac{11}{21}, p_2 =\frac{1}{3} , p_3 = \frac{1}{7}
$$

接下来看计算机数值解

## 转移概率矩阵
P = np.array([[0.5,0.4,0.1],[0.7,0.2,0.1],[0.2,0.4,0.4]])
## 初始状态
P_0 = [0.5,0.4,0.1]
##
for i in range(10):
    P_0 = np.dot(P_0,P)
    print(P_0)

最后稳定在$[p_1,p_2,p_3] = [0.524,0.333,0.143]$,与解析结果一致。

### 第二题

属于$S$ 类的人们中，其第三代将接受高等教育的概率

$$
p = 0.4 \times 0.1 + 0.4 \times 0.2 + 0.2 \times 0.7 = 0.26
$$

如果父母之一受过高等教育，那么他们的子女总可以进入大学，上面的转移矩阵修改为


$$
P=\left[\begin{array}{cccc}
{0.7} & {0.2} & {0.1}  \\ 
{0.4} & {0.4} & {0.2} \\ 
{0} & {0} & {1}.
\end{array}\right]
$$

假设以后代有95%的概率进入大学为结束标准

## 状态转移矩阵
P = np.array([[0.7,0.2,0.1],[0.4,0.4,0.2],[0,0,1]])
P_0 = [0.7,0.2,0.1]
k = 1
while P_0[2] < 0.95:
    P_0 = np.dot(P_0,P)
    print(P_0)
    k = k +1
print('E需要',k,'代')

## 状态转移矩阵
P = np.array([[0.7,0.2,0.1],[0.4,0.4,0.2],[0,0,1]])
P_0 = [0.4,0.4,0.2]
k = 1
while P_0[2] < 0.95:
    P_0 = np.dot(P_0,P)
    print(P_0)
    k = k +1
print('S需要',k,'代')

## 状态转移矩阵
P = np.array([[0.7,0.2,0.1],[0.4,0.4,0.2],[0,0,1]])
P_0 = [0,0,1]
k = 1
while P_0[2] < 0.95:
    P_0 = np.dot(P_0,P)
    print(P_0)
    k = k +1
print('C需要',k,'代')





