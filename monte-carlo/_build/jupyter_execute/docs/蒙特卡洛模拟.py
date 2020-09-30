# 数学建模高阶课:蒙特卡洛模拟

# 蒙特卡洛模拟简介

**蒙特卡罗（Monte Carlo）模拟**其实是对**一种思想的泛指**，只要在解决问题时，利用大量随机样本，然后对这些样本进行概率分析，从而来预测结果的方法，都可以称为蒙特卡洛方法。

蒙特卡罗模拟因摩纳哥著名的赌场而得名。它能够帮助人们从数学上表述物理、化学、工程、经济学以及环境动力学中一些非常复杂的相互作用。

***

# 问题引入

例 1 如何计算圆周率$\pi$

例 2 如何计算定积分$\theta=\int_{0}^{1} x^{2} d x$和$\theta=\int_{2}^{4} e^{-x} d x$

例3 求解整数规划 $Max$ $f=x+y+z$ 约束条件$x^2+y^2+z^2\leq10000$

# 数学原理

## 基本思想

蒙特卡洛模拟通过抓住事件的特征，利用数学方法进行模拟，是一种数字模拟实验。它是一个以概率模型为基础，按照这个模型所描绘的过程，通过模拟实验的结果，作为问题的近似解。

当所求解问题是某种随机事件出现的概率，或者是某个随机变量的期望值时，通过某种“实验”的方法，以这种事件出现的频率估计这一随机事件的概率，或者得到这个随机变量的某些数字特征，并将其作为问题的解。

## 建模方法

**（1）构造或描述概率过程**
对于本身就具有随机性质的问题，如求解圆周率问题，主要是正确描述和模拟这个概率过程，对于本来不是随机性质的确定性问题，比如计算定积分，就必须事先构造一个人为的概率过程，它的某些参量正好是所要求问题的解。即要将不具有随机性质的问题转化为随机性质的问题。
**（2）进行随机模拟**
实现从已知概率分布抽样构造了概率模型以后，由于各种概率模型都可以看作是由各种各样的概率分布构成的，因此产生已知概率分布的随机变量（或随机向量），就成为实现蒙特卡罗方法模拟实验的基本手段，这也是蒙特卡罗方法被称为随机抽样的原因。

在计算机上，可以用物理方法产生随机数，但价格昂贵，不能重复，使用不便。另一种方法是用数学递推公式产生。这样产生的序列，与真正的随机数序列不同，所以称为伪随机数，或伪随机数序列。不过，经过多种统计检验表明，它与真正的随机数，或随机数序列具有相近的性质，因此可把它作为真正的随机数来使用。随机数是我们实现蒙特卡罗模拟的基本工具。

**（3）建立各种估计量**
一般说来，构造了概率模型并能从中抽样后，即实现模拟实验后，我们就要确定一个随机变量，作为所要求的问题的解，我们称它为无偏估计。建立各种估计量，相当于对模拟实验的结果进行考察和登记，从中得到问题的解。

## 数学应用

通常蒙特卡洛方法通过构造符合一定规则的随机数来解决数学上的各种问题。对于那些由于计算过于复杂而难以得到解析解或者根本没有解析解的问题，蒙特卡洛方法是一种有效的求出数值解的方法。蒙特卡洛常见的应用有**蒙特卡洛积分、非线性规划。**

# 问题求解与代码实现

## 1.圆周率求解

### 原理简述

一个正方形内部相切一个圆，圆的面积是$C$，正方形的面积$S$，圆和正方形的面积之比是$\pi/4$
$$
\frac{C}{S}=\frac{\pi r^{2}}{4 r^{2}}=\frac{\pi}{4}
$$

在这个正方形内部，随机产生$n$个点（这些点服从均匀分布），计算它们与中心点的距离是否大于圆的半径，以此判断是否落在圆的内部。落在圆内部的点数统计出来是$m$个点。那么$m、n$点数个数的比例也符合面积的比：
$$
\frac{m}{n}=\frac{\pi}{4}
$$
$m$与$n$的比值乘以4，就是$\pi$的值:
$$
\pi=\frac{m}{n} \cdot 4
$$
前提是$m$、$n$足够大的话。

### 代码实现

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
%matplotlib inline

np.random.seed(1)

n = 10000
r = 1.0
a,b = 0.0,0.0
xmin, xmax = a-r, a+r
ymin, ymax = b-r, b+r

#随机生成n=1000个点（重要，python随机生成一定范围内一定大小数组）
x = np.random.uniform(xmin,xmax,n)
y = np.random.uniform(ymin,ymax,n)


fig = plt.figure(figsize=(6,6))
axes = fig.add_subplot(1,1,1)#添加子图
#画子图
plt.plot(x,y,'ko',markersize = 1) #plot绘图 markersize表示点的大小；‘ro’r表示red，o表示圆圈
plt.axis('equal') #表示x轴和y轴的单位长度相同

#求点到圆心的距离
d = np.sqrt((x-a)**2 + (y-a)**2)

#res 得到圆中的点数
res = sum(np.where(d<r,1,0)) #numpy.where(conditon,x,y) 满足条件输出x，不满足输出y

pi = res/n*4
print('pi:',pi)
#计算pi的近似值，蒙特卡洛模拟方法，用统计值去近似真实值

#绘制圆形子图
circle = Circle(xy = (a,b), radius = r,alpha = 0.5, color = 'gray')
axes.add_patch(circle)#添加圆形子图
plt.grid(True,linestyle = '--',linewidth = 0.8)
plt.show()

#蒙特卡洛模拟是用统计值逼近真实值，展示了统计思想

## 2.定积分求解

### 原理简述

考虑估计$\theta=\int_{0}^{1} g(x) d x$，若$X_{1}, \cdots, X_{m}$为均匀分布$U(0,1)$总抽取的样本，则由强大数定律知
$$
\hat{\theta}=\overline{g_{m}(X)}=\frac{1}{m} \sum_{i=1}^{m} g\left(X_{i}\right)
$$
以概率1收敛到期望$E g(X)$，因此$\theta=\int_{0}^{1} g(x) d x$的简单的Monte Carlo 估计量为$\overline{g_{m}(X)}$

### 求解定积分
$$
\theta=\int_{0}^{1} x^{2} d x
$$

m=100000  #要确保m足够大

Sum=0
import random
for i in range(m):
    x = random.random()        #返回随机生成的一个实数，它在[0,1)范围内。     
    y = x**2
    Sum+=y
 
R=Sum/m
 
print(R)

### 投点法求解（类似求圆周率）

n=100000   #n足够大
 
m=0                                  
 
import random
 
for i in range(n):
    x = random.random()
    y = random.random()
    if x**2>y:                       #表示该点位于曲线y=x^2的下面
        m=m+1
 
R=m/n
 
print(R)

### 进一步推广

若要计算$\int_{a}^{b} g(x) d x$，此处$a<b$，则作一积分变换使得积分限从0到1，即做变换$y=(x-a) /(b-a)$，因此
$$
\int_{a}^{b} g(x) d x=\int_{0}^{1} g(y(b-a)+a)(b-a) d y
$$

## 整数规划求解

问题回顾:求解整数规划 $\max f=x+y+z$ 约束条件$x^2+y^2+z^2\leq10000$

### 原理简述

由均值不等式
$$
\begin{array}{l}{H_{n} \leqslant G_{n} \leqslant A_{n} \leqslant Q_{n}} \\ {\dfrac{n}{\sum_{i=1}^{n} \frac{1}{x_{i}}} \leqslant \sqrt[n]{\prod_{i=1}^{n} x_{i}} \leqslant \dfrac{\sum_{i=1}^{n} x_{i}}{n} \leqslant \sqrt{\dfrac{\sum_{i=1}^{n} x_{i}^{2}}{n}}}\end{array}
$$
可知
$$
\frac{x+y+z}{3} \leq \sqrt{\frac{x^{2}+y^{2}+z^{2}}{3}}=\sqrt{\frac{10000}{3}}
$$
即
$$
x+y+z =\sqrt{30000}\approx173.2
$$

由于这个问题是整数规划问题，上式取最大值时$x=y=z=\sqrt{\frac{10000}{3}}$不满足要求，同时使用多元函数求导的办法也得不到最优整数解

但整数解是有限个，于是为枚举法提供了方便。

如果用显枚举法试探，共需计算 $(100)^3 = 10^6$个点，其计算量较大。然而应用蒙特卡洛去随机计算$10^4$个点，便可找到满意解，那么这种方法的可信度究竟怎样呢？

不失一般性，假定一个整数规划的最优点不是孤立的奇点。

假设目标函数落在高值区的概率分别为 $0.01$，$0.001$，则当计算$10^4$个点后，有任意一个点落在高值区的概率分别为

$$
\begin{array}{l}{1-0.99^{10000} \approx 0.99 \cdots 99(超过10位)} \\ {1-0.999^{10000} \approx 0.9999548267}\end{array}
$$

### 代码求解

m=100000  #要确保m足够大
maxf=0

import random
 
for i in range(m):
    while True :
        x = random.randint(0,100)      
        y = random.randint(0,100) 
        z = random.randint(0,100) 
        if x**2+y**2+z**2<=10000 :
            break
    max=x+y+z
    if max>maxf :
        maxf=max
        xmax=x
        ymax=y
        zmax=z

print('maxf:',maxf,'xmax:',xmax,'ymax:',ymax,'zmax:',zmax)

结果和$\sqrt{30000}\approx173.2$相近，说明蒙特卡罗模拟可以得到一个满意解（一般可以得到最优解173）

# 蒙特卡洛积分法

## 蒙特卡洛法的性质

###  收敛性

前面所述告诉我们，蒙特卡罗积分法是以随机变量$Y_{1}, Y_{2}, \cdots, Y_{N}$的简单抽样的算术平均值

$\bar{Y}_{N}=\frac{1}{N} \sum_{i=1}^{N} Y_{i}$

作为积分的近似值（忽略休积因子）。因为$Y_{1}, Y_{2}, \cdots, Y_{N}$独立同分布，期望有限，则根据强大数定律．我们有 

$P\left(\lim _{N \rightarrow \infty} \bar{X}_{N}=\mathbf{E}(X)\right)=1$

即当抽样数N充分大时，随机变最的简单抽样的算术平均位以概率1收敛到它的期望值。这意味着．样本平均值法计算的积分值以概率1收敛于积分的真值。

### 误差大小

关于蒙特卡罗方法的近似值与真位的误差问题，我们需要运用中心极限定理来分析做出结论。

中心极限定理指出，如果随机变呈序列$Y_{1}, Y_{2}, \cdots, Y_{N}$独立同分布且方差有限，则有

$\lim _{N \rightarrow \infty} P\left(\frac{\sqrt{N}}{v}\left|\bar{Y}_{N}-\mathrm{E}(Y)\right|<\delta\right)=\frac{1}{\sqrt{2 \pi}} \int_{-\delta}^{\delta} \mathrm{e}^{-t^{2} / 2} \mathrm{d} t$

其中 $\delta>0$ 当N充分大时，就有如下的近似式： 

$P\left(\left|\bar{Y}_{N}-\mathbf{E}(Y)\right|<\frac{z_{\alpha / 2} v}{\sqrt{N}}\right) \approx \frac{2}{\sqrt{2 \pi}} \int_{0}^{z_{\alpha / 2}} \mathrm{e}^{-t^{2} / 2} \mathrm{d} t=1-\alpha$

其中$\alpha$是显著水平（(1一$\alpha$)为置信度）,$z_{\alpha / 2}$是标准正态分布在显著水平为$\alpha / 2$的临界 值，$v$是$Y$的标准差。上式表明，如下不等式

$\left|\bar{Y}_{N}-\mathbf{E}(Y)\right|<\frac{z_{\alpha / 2} v}{\sqrt{N}}$

近似的以概率(1一$\alpha$)成立，且误差的收敛速度为$O\left(N^{-1 / 2}\right)$。
于是，蒙特卡洛积分法的误差定义为：
$\varepsilon(\hat{I})=\frac{z_{\alpha / 2} v}{\sqrt{N}}$

关于蒙特卡罗积分法的误差需说明两点：第一，蒙特卡罗积分法的误差为概率意义上的误差，这与其他数值计算方法是有区别的；第二，误差中的标准差是未知的，必须使用其估计值。

$\hat{v}=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(Y_{i}-\bar{Y}_{N}\right)^{2}}$

### 优点

#### 受几何条件限制小。

在计算高维空问中的某一区域D上的重积分时，无论积分区域D的形状多么特殊，只要能给出描述的几何特征的条件，就可以从D中均匀产生N个点，从而获得该积分的近似值。这是一般数值计算方法难以做到的。另外在具有随机性质的问题中 ，如考虑的系统形状很复杂，难以用一般数值方法求解，而使用 蒙特卡罗积分法则不会有原则上的困难。 

#### 收敛速度与问题的维数无关。

从误差分析可知，在给定置信水平的情况下，蒙特卡罗积分法的收敛速度为$O\left(N^{-1 / 2}\right)$与问题木身的维数无关，这是一般数值方法计算高维积分时难以克服的问题。

#### 具有同时计算多个积分量的能力。 

对于某些物理问题中需要计算多个积分量的情况，它们可以在一个蒙特卡罗积分程 序中同时完成计算，不需要像常规方法那样逐一地编程计算，即可以共享随机数序列。

#### 误差容易确定。 

对于一般计算方法，要给出计算结果与真值的误差并不是一件容易的事情，而蒙特卡罗方法则不然。根据蒙特卡罗方法的误差公式 即使对于很复杂的积分计算问题，也是一样的，在计算所求积分的同时可估计出误差。

#### 程序非常简单，易于编程实现。 

### 方法的缺点

#### 收敛速度慢。 
如前所述，蒙特卡罗方法的收敛速度为精确度较高的近似结果一般不容易得到。
（三维以下）的问题，它不如其他数值方法好。

#### 误差具有概率性。 
由于蒙特卡罗积分法的误差是在一定置信度下估计的．所以它的误差具有概率性，而不是一般意义的误差。

# 数学建模实例

## 问题简述

每逢节假日，年轻人们都喜欢去电影院看电影。看电影的过程往往是愉快的，但是当我们从观影厅出来上厕所时，厕所门口老是排起了一条长队。怎么样才能解决这个问题？

## 作出假设

**tip:**下面模型作了很多强假设，建模的过程往往就是一个从简单到复杂的过程

1.两场电影结束时间相隔较长，互不影响； 

2.每场电影结束之后会有20个人想上洗手间；

3.这20个人会在0到10分钟之内全部到达洗手间（第3,4条使用了均匀分布，更严谨的做法是使用正态分布）；

4.每个人上洗手间时间在1-3分钟之间。

5.洗手间只有一个位置，不考虑两人共用的情况即每人必须等上一人离开后方可进入。

## 问题分析

对于每个人都有如下几个参数：

1.到达时间

2.等待时间

3.开始上洗手间时间

4.结束时间

![Image Name](https://cdn.kesci.com/upload/image/q31st23kmm.png?imageView2/0/w/960/h/960)

## 代码实现

arrivingtime = np.random.uniform(0,10,size = 20)
arrivingtime.sort()
arrivingtime

import numpy as np


#首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和上厕所耗时：
#函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
arrivingtime = np.random.uniform(0,10,size = 20)
#.sort() 降序排列
arrivingtime.sort()
#上厕所时间初始化
working = np.random.uniform(1,3,size = 20)

#到达时间、结束时间、厕所空闲时间时间、等待时间初始化
startingtime = [0 for i in range(20)]
finishtime = [0 for i in range(20)]
waitingtime = [0 for i in range(20)]
emptytime = [0 for i in range(20)]

#对第一个人的情况单独处理
startingtime[0] = arrivingtime[0]
finishtime[0] = startingtime[0] + working[0]
waitingtime[0] = startingtime[0]-arrivingtime[0]

#第二个以后用循环
        
for i in range(1,len(arrivingtime)):
    if finishtime[i-1] > arrivingtime[i]:
        startingtime[i] = finishtime[i-1] # 你的理解
    else:
        startingtime[i] = arrivingtime[i]
        emptytime[i] = startingtime[i] - finishtime[i-1]
    finishtime[i] = startingtime[i] + working[i]
    waitingtime[i] = startingtime[i] - arrivingtime[i]
    
    #print(waitingtime[i])
print("average waiting time is %f" % np.mean(waitingtime))

### 结论一:随机模拟下平均每人等待时间十几分钟

Now: 改写了程序加入循环，求重复实验下每人平均等待时间

import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和上厕所耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    arrivingtime = np.random.uniform(0,10,size=20)
    #.sort() 降序排列
    arrivingtime.sort()
    #上厕所时间
    working = np.random.uniform(1,3,size=20)

    startingtime = [0 for i in range(20)]
    finishtime = [0 for i in range(20)]
    waitingtime = [0 for i in range(20)]
    emptytime = [0 for i in range(20)]

    #对第一个人的情况单独处理
    startingtime[0] = arrivingtime[0]
    finishtime[0] = startingtime[0] + working[0]
    waitingtime[0] = startingtime[0]-arrivingtime[0]

    #第二个以后用循环
    
    #计算一下每人的等待时间：
    for i in range(1,len(arrivingtime)):
        if finishtime[i-1] > arrivingtime[i]:
            startingtime[i] = finishtime[i-1]
        else:
            startingtime[i] = arrivingtime[i]
            emptytime[i] = startingtime[i] - finishtime[i-1]
        finishtime[i] = startingtime[i] + working[i]
        waitingtime[i] = startingtime[i] - arrivingtime[i]
        #print(waitingtime[i])
    #print("average waiting time is %f" % np.mean(waitingtime))
    return np.mean(waitingtime)
sum = 0
for i in range(1000):
        sum+=forecast();
avg_waitingtime=sum/1000
print("average waiting time is %f" %avg_waitingtime )

### 结论二:随机模拟重复实验下每人平均等待时间约为14.5分钟

上个厕所要等待14.5分钟,这种情况显然必须得到改变,那我们要怎么改进呢?

### 改进策略:增加一个厕所位置

import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和上厕所耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    arrivingtime = np.random.uniform(0,10,size=20)
    #.sort() 降序排列
    arrivingtime.sort()
    #上厕所时间
    working = np.random.uniform(1,3,size=20)

    startingtime = [0 for i in range(20)]
    finishtime = [0 for i in range(20)]
    waitingtime = [0 for i in range(20)]
    emptytime = [0 for i in range(20)]

    #对第一个人的情况单独处理
    startingtime[0] = arrivingtime[0]
    finishtime[0] = startingtime[0] + working[0]
    waitingtime[0] = startingtime[0] - arrivingtime[0]
    
    #对第二个人的情况单独处理
    startingtime[1] = arrivingtime[1]
    finishtime[1] = startingtime[1] + working[1]
    waitingtime[1] = startingtime[1] - arrivingtime[1]
    for i in range(2,len(arrivingtime)):
        if finishtime[i-1] > arrivingtime[i] and finishtime[i-2] > arrivingtime[i]:
            startingtime[i] = min(finishtime[i-1],finishtime[i-2])
        else:
            startingtime[i] = arrivingtime[i]
            emptytime[i] = startingtime[i] - finishtime[i-1]

        finishtime[i] = startingtime[i] + working[i]
        waitingtime[i] = startingtime[i] - arrivingtime[i]
        #print(waitingtime[i])
    #print("average waiting time is %f" % np.mean(waitingtime))
    return np.mean(waitingtime)
Sum = 0
for i in range(1000):
        Sum+=forecast()
avg_waitingtime=Sum/1000
print("average waiting time is %f" %avg_waitingtime )

### 结论3:增加一个位置明显使等待时间变短

显然增加厕所数量是可行的，但厕所拥堵应该还和短时间（看完电影）大量的人要上厕所有关，我们来看看是不是这样

import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和上厕所耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    
    arrivingtime = np.random.uniform(0,3,size=20)   #假设这20个人在前3分钟到达
    
    #.sort() 降序排列
    arrivingtime.sort()
    #上厕所时间
    working = np.random.uniform(1,3,size=20)

    startingtime = [0 for i in range(20)]
    finishtime = [0 for i in range(20)]
    waitingtime = [0 for i in range(20)]
    emptytime = [0 for i in range(20)]

    #对第一个人的情况单独处理
    startingtime[0] = arrivingtime[0]
    finishtime[0] = startingtime[0] + working[0]
    waitingtime[0] = startingtime[0] - arrivingtime[0]
    
    #对第二个人的情况单独处理
    startingtime[1] = arrivingtime[1]
    finishtime[1] = startingtime[1] + working[1]
    waitingtime[1] = startingtime[1] - arrivingtime[1]
    for i in range(2,len(arrivingtime)):
        if finishtime[i-1] > arrivingtime[i] and finishtime[i-2] > arrivingtime[i]:
            startingtime[i] = min(finishtime[i-1],finishtime[i-2])
        else:
            startingtime[i] = arrivingtime[i]
            emptytime[i] = startingtime[i] - finishtime[i-1]

        finishtime[i] = startingtime[i] + working[i]
        waitingtime[i] = startingtime[i] - arrivingtime[i]
        #print(waitingtime[i])
    #print("average waiting time is %f" % np.mean(waitingtime))
    return np.mean(waitingtime)
Sum = 0
for i in range(1000):
        Sum+=forecast();
avg_waitingtime=Sum/1000
print("average waiting time is %f" %avg_waitingtime )

### 结论4：集中在放映结束后上厕所使等待时间大幅增加，所以看完电影不妨把彩蛋看完

### 模型改进点

1.到达时间和上洗手间耗时不可能是均匀分布，应该按正态分布更加合理；

2.没有考虑上大号的情况hhh；

3.把上厕所的总人数定在了20，可以用随机数模拟上厕所的总人数。

## 优缺点分析

蒙特卡洛模拟的特点：随机采样得到的近似解，随着随机采样数值增多，得到正确结果的概率越大

借助计算机技术，蒙特卡洛模拟方法有两大优点
- 简单，省去了繁复的数学推导和验算过程，使普通人能够理解
- 快速，建模过程简单，确定了概率模型，后续运算完全用计算机实现

蒙特卡洛模拟方法有存在一定的缺陷
- 如果必须输入一个模式中的随机数并不像设想的那样是随机数， 而却构成一些微妙的非随机模式， 那么整个的模拟（及其预测结果）都可能是错的。

贝尔实验室的里德博士告诫人们记住伟大的诺伊曼的忠告:“任何人如果相信计算机能够产生出真正的随机的数序组都是疯子。”

