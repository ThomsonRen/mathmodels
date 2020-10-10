# 模拟退火算法


模拟退火算法 (Simulated Annealing，SA) 最早的思想是由 N. Metropolis 等人于1953年提出。1983年, S. Kirkpatrick 等成功地将退火思想引入到组合优化领域。它是**基于 Monte-Carlo 迭代求解策略的一种随机寻优算法**，其出发点是基于物理中固体物质的退火过程与一般组合优化问题之间的相似性。模拟退火算法从某一较高初温出发，伴随温度参数的不断下降,结合概率突跳特性在解空间中随机寻找目标函数的全局最优解，即在局部最优解能概率性地跳出并最终趋于全局最优。

与具有陷入局部最小值的缺点的基于梯度的方法(也就是刚才提到的爬山法)以及其他确定性搜索方法不同，SA 的主要优势在于其具有避免陷入局部最小值的能力。事实上，已经证明，如果足够的随机性与非常缓慢的冷却结合使用，模拟退火将收敛到其全局最优性。实质上，SA 是一种搜索算法，**可以视为马尔可夫链**，能够在合适的条件下收敛。

模拟退火在建模比赛中最主要的应用是模型求解，特别是最优化问题（组合优化问题），如旅行商问题 ( TSP）、最大截问题(Max Cut Problem)、0-1背包问题(Zero OneKnapsack Problem)、图着色问题(Graph Colouring Problem)、调度问题(Scheduling Problem)等等。


## 物理学中的退火


**加温过程**：增强粒子运动，消除系统原先可能存在的非均匀态。

**等温过程**：对于与环境换热而温度不变的封闭系统，系统状态的自发变化总是朝自由能减少的方向进行，当自由能达到最小时，系统达到平衡。

**冷却过程**：使粒子热运动减弱并渐趋有序，系统能量逐渐下降，从而得到低能的晶体结构。



## Boltzman 概率分布



在统计学中，玻尔兹曼分布的表达形式为

$$
F(state) \propto  e^{-E/kT}
$$
上式中，$E$是能量状态，$kT$是玻尔兹曼常数和热力学温度的乘积。

系统的两种状态之间的玻尔兹曼分布比率称为玻尔兹曼因子，

$$
\dfrac{F(state1) }{F(state2) } = e^{-(E_1 - E_2 )/kT}
$$


- 同一个温度，分子停留在能量小状态的概率大于停留在能量大状态的概率
- 温度越高，不同能量状态对应的概率相差越小，温度足够高时，各状态对应概率基本相同。
- 随着温度的下降，能量最低状态对应概率越来越大，温度趋于0时，其状态趋于1


## 劣解接受概率（Metropolis准则）

劣解接受概率又称做状态接受函数，这里是模拟退火法这个名字的来源，模仿固体退火原理，随着温度的下降（迭代次数上升），能量逐渐稳定。即劣解的接受概率$p$逐渐下降，其公式为：

$$
\displaystyle
p=\left\{\begin{array}{ll}{1,} & {E\left(x_{new}\right) < E\left(x_{o l d}\right)} \\ {\exp \left({-\dfrac{E\left(x_{n e w}\right)-E\left(x_{o l d}\right)}{T}}\right),} & {E\left(x_{n e w}\right) \geq E\left(x_{o l d}\right)}\end{array}\right.
$$



从公式中可以看出这是求最小值时的分布，因为当新的解小于旧解时百分百接受。又可以看出当$E\left(x_{n e w}\right) \geq E\left(x_{o l d}\right)$时，$p$恒在[0, 1]之间。可以看到，随着时间推移，温度越低，对于劣解的接受概率越低。

import numpy as np
import matplotlib.pyplot as plt
deltaEList = [10,20,30,40,50]
for deltaE in deltaEList:
    p_list = []
    t_list = [i for i in range(25,100)]
    for T in range(25,100):
        p_list.append(np.exp(- deltaE/T))
    plt.plot(t_list,p_list,label = r'$\Delta E$ = ' + str(deltaE))
plt.legend()
plt.xlabel(r'$T$')
plt.ylabel(r'$p$')

## 温度初始化和更新

### 温度初始化
关于初始温度，初温越大，获得高质量解的机率越大，但花费较多的计算时间，原理可以从上面的图中或者结合下面温度随时间变化进行理解、解释，温度越高对于劣解的接受能力越大（搜索范围越大），下降速度越慢（求解速度下降）。

关于初始解如何设定一般有以下方法：

（1）按照某一概率分布抽样一组$K$个解，以$K$个解的目标值的方差为初温（下面使用的时正态分布）。

（2）随机产生一组$K$个状态，确定两两状态间的最大目标差值，根据差值，利用一定的函数确定初温。

（3）利用经验公式。



### 温度更新

####  更新函数1:对数函数

$$
T(t+1)=\frac{T_{0}}{\ln (1+t)}
$$

如果使用这种温度更新，不管初始温度如何，都有一个先加热后散热的过程。即先提高搜索范围，再慢慢的减少搜索范围。

T0_list = [1000,500,200,100]
for T0 in T0_list:
    T_list = []
    t_list = [i for i in range(1,40)]
    T_list.append(T0)
    for t in t_list:
        T_list.append(T0/np.log(t+1))
    plt.plot([i for i in range(40)],T_list,label = 'T0 = '+ str(T0))
plt.legend()

####   线性函数

$$
T(t+1)= \alpha T(t), \alpha \in (0,1)
$$

指数衰减，通常$0.8 < \alpha <0.99$。

T0_list = [1000,500,200,100]
alpha = 0.99
for T0 in T0_list:
    T_list = []
    t_list = [i for i in range(1,400)]
    T_list.append(T0)
    for t in t_list:
        T_list.append(alpha * T_list[-1] )
    plt.plot([i for i in range(400)],T_list,label = 'T0 = '+ str(T0))
plt.legend()

####  更新函数3：反比例函数

$$
T(t)=  \frac{T_{0}}{1+t}
$$

t是迭代次数。按照反比例函数衰减，类似更新函数2。

T0_list = [1000,500,200,100]
alpha = 0.99
for T0 in T0_list:
    T_list = []
    t_list = [i for i in range(1,40)]
    T_list.append(T0)
    for t in t_list:
        T_list.append(T0/(t+1))
    plt.plot([i for i in range(40)],T_list,label = 'T0 = '+ str(T0))
plt.legend()

## 算法步骤


![Image Name](https://cdn.kesci.com/upload/image/q3w7sz5x3e.png?imageView2/0/w/960/h/960)

**步骤**

1. 初始化温度$T$，初始解状态$S$，每个温度$t$下的迭代次数$L$；

2. 当$k = 1,2,\cdots,L$时，进行3~6；

3. 对当前解进行变换得到新解$S'$（例如对某些解中的元素进行互换，置换）；

4. 计算增量$\Delta t'=C(S')-C(S)$，其中$C(S)$为评价函数；

5. 若$\Delta t'<0$则接受$S'$作为新的当前解，否则以概率$e^{-\Delta t'/(KT)}$接受$S′$作为新的当前解($K$为玻尔兹曼常数，数值为：$K=1.380650× 10^{-23} J/K$）；（实际上，为了简便运算，而又不失一般性，通常忽略$K$，令$K=1$）

6. 如果满足终止条件则输出当前解作为最优解，结束程序；

7. 减小$T$，转到第2步，直到$T$小于初始设定的阈值。



## 小结

模拟退火其实也是一种Greedy算法，但是它的搜索过程引入了随机因素。**模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解。**以上图为例，模拟退火算法在搜索到局部最优解B后，会以一定的概率接受向右继续移动。也许经过几次这样的不是局部最优的移动后会到达B 和C之间的峰点，于是就跳出了局部最小值B。



![Image Name](https://cdn.kesci.com/upload/image/q3w7vbb61r.png?imageView2/0/w/960/h/960)



模拟退火算法来源于固体退火原理，将固体加温至充分高，再让其徐徐冷却，加温时，固体内部粒子随温升变为无序状，内能增大，而徐徐冷却时粒子渐趋有序，在每个温度都达到平衡态，最后在常温时达到基态，内能减为最小。

根据 Metropolis 准则，粒子在温度$T$时趋于平衡的概率为$e^{-\frac{\Delta E}{kT}}$，其中$E$为温度$T$时的内能，$\Delta E$为其改变量，$k$为 Boltzmann 常数。

用固体退火模拟组合优化问题，将内能$E$模拟为目标函数值$f$，温度$T$演化成控制参数$t$，即得到解组合优化问题的模拟退火算法：

由初始解i和控制参数初值t开始，对当前解重复“产生新解→计算目标函数差→接受或舍弃”的迭代，并逐步衰减$t$值，算法终止时的当前解即为所得近似最优解，这是基于蒙特卡罗迭代求解法的一种启发式随机搜索过程。退火过程由冷却进度表(Cooling Schedule)控制，包括控制参数的初值$t$及其衰减因子$\Delta t$、每个$t$值时的迭代次数$L$和停止条件$S$。


让我们实现例1、2的模拟退火算法，特别关注：
  - 初始化方法、
  - 温度更新方法、
  - 产生新解方法


## 例1代码实现：函数寻优（最小值）

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
 
#define aim function
def aimFunction(x):
    y=x**3-60*x**2-4*x+6
    return y
x=[i/10 for i in range(1000)]
y=[0 for i in range(1000)]
for i in range(1000):
    y[i]=aimFunction(x[i])
 
plt.plot(x,y)

T=1000 #initiate temperature
Tmin=10 #minimum value of terperature
x=np.random.uniform(low=0,high=100)#initiate x
k=50 #times of internal circulation 
y=0#initiate result
t=0#time
while T>=Tmin:
    for i in range(k):
        #calculate y
        y=aimFunction(x)
        #generate a new x in the neighboorhood of x by transform function
        xNew=x+np.random.uniform(low=-0.055,high=0.055)*T
        if (0<=xNew and xNew<=100):
            yNew=aimFunction(xNew)
            if yNew-y<0:
                x=xNew
            else:
                #metropolis principle
                p=math.exp(-(yNew-y)/T)
                r=np.random.uniform(low=0,high=1)
                if r<p:
                    x=xNew
    t+=1
#     print (t)
    T=1000/(1+t)
    
print ('x:',x,'|y:',aimFunction(x))

## 例2代码实现：求解TSP

num = 52
solutionnew = np.arange(num)
solutionnew

import numpy as np
import matplotlib.pyplot as plt 
import pdb
 
"旅行商问题 ( TSP , Traveling Salesman Problem )"
coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                        [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                        [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                        [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                        [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                        [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
                        [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
                        [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
                        [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
                        [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
                        [1340.0,725.0],[1740.0,245.0]])
 
#得到距离矩阵的函数
def getdistmat(coordinates):
    num = coordinates.shape[0] #52个坐标点
    distmat = np.zeros((52,52)) #52X52距离矩阵
    for i in range(num):
        for j in range(i,num):
            distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])
    return distmat
 
def initpara():
    alpha = 0.99
    t = (1,100)
    markovlen = 1000
 
    return alpha,t,markovlen
num = coordinates.shape[0]
distmat = getdistmat(coordinates) #得到距离矩阵
 
 
solutionnew = np.arange(num)
#valuenew = np.max(num)
 
solutioncurrent = solutionnew.copy()
valuecurrent =99000  #np.max这样的源代码可能同样是因为版本问题被当做函数不能正确使用，应取一个较大值作为初始值
#print(valuecurrent)
 
solutionbest = solutionnew.copy()
valuebest = 99000 #np.max
 
alpha,t2,markovlen = initpara()
t = t2[1]
 
result = [] #记录迭代过程中的最优解
while t > t2[0]:
    for i in np.arange(markovlen):
 
        #下面的两交换和三交换是两种扰动方式，用于产生新解
        if np.random.rand() > 0.5:# 交换路径中的这2个节点的顺序
            # np.random.rand()产生[0, 1)区间的均匀随机数
            while True:#产生两个不同的随机数
                loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                ## print(loc1,loc2)
                if loc1 != loc2:
                    break
            solutionnew[loc1],solutionnew[loc2] = solutionnew[loc2],solutionnew[loc1]
        else: #三交换
            while True:
                loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(num-1))) 
                loc3 = np.int(np.ceil(np.random.rand()*(num-1)))
 
                if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                    break
 
            # 下面的三个判断语句使得loc1<loc2<loc3
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1
            if loc2 > loc3:
                loc2,loc3 = loc3,loc2
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1
 
            #下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
            tmplist = solutionnew[loc1:loc2].copy()
            solutionnew[loc1:loc3-loc2+1+loc1] = solutionnew[loc2:loc3+1].copy()
            solutionnew[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()  
 
        valuenew = 0
        for i in range(num-1):
            valuenew += distmat[solutionnew[i]][solutionnew[i+1]]
        valuenew += distmat[solutionnew[0]][solutionnew[51]]
       # print (valuenew)
        if valuenew<valuecurrent: #接受该解
           
            #更新solutioncurrent 和solutionbest
            valuecurrent = valuenew
            solutioncurrent = solutionnew.copy()
 
            if valuenew < valuebest:
                valuebest = valuenew
                solutionbest = solutionnew.copy()
        else:#按一定的概率接受该解
            if np.random.rand() < np.exp(-(valuenew-valuecurrent)/t):
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()
            else:
                solutionnew = solutioncurrent.copy()
    t = alpha*t
    result.append(valuebest)
    # print (t) #程序运行时间较长，打印t来监视程序进展速度
#用来显示结果
plt.plot(np.array(result))
plt.ylabel("bestvalue")
plt.xlabel("t")
plt.show()

## 模拟退火算法总结


- 迭代搜索效率高，并且可以并行化；
- 算法中有一定概率接受比当前解较差的解，因此一定程度上可以跳出局部最优；
- 算法求得的解与初始解状态S无关，因此有一定的鲁棒性；
- 具有渐近收敛性，已在理论上被证明是一种以概率l收敛于全局最优解的全局优化算法。