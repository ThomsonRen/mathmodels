---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "A121262452184A35B0EFBCD4BF0997B6", "mdEditEnable": false, "jupyter": {}}

# 蒙特卡洛模拟


+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "D74168070CA1427E8CFE512AF432FF02", "mdEditEnable": false, "jupyter": {}}

## 蒙特卡洛模拟简介

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "88664AC64646494B8F13733D9998D9C4", "jupyter": {}, "mdEditEnable": false}

**蒙特卡罗（Monte Carlo）模拟**其实是对**一种思想的泛指**，只要在解决问题时，利用大量随机样本，然后对这些样本进行概率分析，从而来预测结果的方法，都可以称为蒙特卡洛方法。


```{figure} ../_static/lecture_specific/monte-carlo/monte-carlo-demo.jpg
---
height: 300px
name: monte-carlo-1
---

```





蒙特卡罗模拟因摩纳哥著名的赌场而得名。它能够帮助人们从数学上表述物理、化学、工程、经济学以及环境动力学中一些非常复杂的相互作用。

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "4934535578F34D8182F911FF3CB50DEF", "jupyter": {}, "mdEditEnable": false}

***

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "14EF9691500B4E24B87298AA2F9E55CE", "jupyter": {}, "mdEditEnable": false}



```{admonition} 问题引入

- 如何计算圆周率$\pi$

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "B7F5D41B3FFD4C4CAD3BA777124A9B0F", "mdEditEnable": false, "jupyter": {}}

- 如何计算定积分

$$
\theta=\int_{0}^{1} x^{2} d x
$$

和

$$
\theta=\int_{2}^{4} e^{-x} d x
$$



 - 求解整数规划 $\max f=x+y+z$ 约束条件$x^2+y^2+z^2\leq10000$

```

## 数学原理



蒙特卡洛模拟通过抓住事件的特征，利用数学方法进行模拟，是一种数字模拟实验。它是一个以概率模型为基础，按照这个模型所描绘的过程，通过模拟实验的结果，作为问题的近似解。

当所求解问题是某种随机事件出现的概率，或者是某个随机变量的期望值时，通过某种“实验”的方法，以这种事件出现的频率估计这一随机事件的概率，或者得到这个随机变量的某些数字特征，并将其作为问题的解。

通常蒙特卡洛方法通过构造符合一定规则的随机数来解决数学上的各种问题。对于那些由于计算过于复杂而难以得到解析解或者根本没有解析解的问题，蒙特卡洛方法是一种有效的求出数值解的方法。蒙特卡洛常见的应用有**蒙特卡洛积分、非线性规划。**





## 案例求解


### 圆周率求解



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


```{code-cell} ipython3
---
id: 81F939604843423580F5871EA298F4E5
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "E5F8BFA6B6D64DAFBBD197F12C02043C", "mdEditEnable": false, "jupyter": {}}

### 定积分求解



考虑估计$\theta=\int_{0}^{1} g(x) d x$，若$X_{1}, \cdots, X_{m}$为均匀分布$U(0,1)$总抽取的样本，则由强大数定律知

$$
\hat{\theta}=\overline{g_{m}(X)}=\frac{1}{m} \sum_{i=1}^{m} g\left(X_{i}\right)
$$

以概率1收敛到期望$E g(X)$，因此$\theta=\int_{0}^{1} g(x) d x$的简单的Monte Carlo 估计量为$\overline{g_{m}(X)}$


```{admonition} 定积分求解
$$
\theta=\int_{0}^{1} x^{2} d x
$$
```


```{code-cell} ipython3
---
id: 502E83533BC14097B5F3E70E1536E935
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
m=100000  #要确保m足够大

Sum=0
import random
for i in range(m):
    x = random.random()        #返回随机生成的一个实数，它在[0,1)范围内。     
    y = x**2
    Sum+=y
 
R=Sum/m
 
print(R)
```



+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "308F4C38F7774D3D8FF12F115C111C1F", "mdEditEnable": false, "jupyter": {}}

除此之外，还可以直接用投点法求解

```{code-cell} ipython3
---
id: 380F78DD156E49948DEECCB9A118AE1E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```


进一步推广, 若要计算$\int_{a}^{b} g(x) d x$，此处$a<b$，则作一积分变换使得积分限从0到1，即做变换$y=(x-a) /(b-a)$，因此

$$
\int_{a}^{b} g(x) d x=\int_{0}^{1} g(y(b-a)+a)(b-a) d y
$$

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "D702B168D6004A178EF77714E7226951", "jupyter": {}, "mdEditEnable": false}

### 整数规划求解


求解整数规划 $\max f=x+y+z$ 

约束条件$x^2+y^2+z^2\leq10000$

首先由均值不等式

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

由于这个问题是整数规划问题，上式取最大值时$x=y=z=\sqrt{\frac{10000}{3}}$不满足要求，同时使用多元函数求导的办法也得不到最优整数解。但整数解是有限个，于是为枚举法提供了方便。

如果用显枚举法试探，共需计算 $(100)^3 = 10^6$个点，其计算量较大。然而应用蒙特卡洛去随机计算$10^4$个点，便可找到满意解，那么这种方法的可信度究竟怎样呢？

不失一般性，假定一个整数规划的最优点不是孤立的奇点。假设目标函数落在高值区的概率分别为 $0.01$，$0.001$，则当计算$10^4$个点后，有任意一个点落在高值区的概率分别为

$$
\begin{array}{l}{1-0.99^{10000} \approx 0.99 \cdots 99(超过10位)} \\ {1-0.999^{10000} \approx 0.9999548267}\end{array}
$$

可以看出，使用蒙特卡洛方法还是比较有把握获得较优解的。下面来看代码实现

```{code-cell} ipython3
---
id: 88B7E2C65B904B15853156AF14A31E7C
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

结果和$\sqrt{30000}\approx173.2$相近，说明蒙特卡罗模拟可以得到一个满意解。



## 数学建模实例


接下来，我们用蒙特卡洛模拟来研究餐厅的排队现象。


```{figure} ../_static/lecture_specific/monte-carlo/canting.jpg
---
height: 300px
name: canting
---

```

首先我们通过一系列假设简化这个具体问题，降低其计算的难度，数学建模的过程往往就是一个从简单到复杂的过程。


```{admonition} 假设
1.  我们研究20个学生进来吃饭的情况
1.  这20个同学会在0到10分钟之内全部到达餐厅
1.  每个人点餐和取餐的用时在1-3分钟之间（第3,4条使用了均匀分布，更严谨的做法是使用正态分布或者泊松分布）
1.  餐厅目前只有一个柜台，每位同学必须等上一人离开后方可点餐
```


对于每个人都有如下几个参数：`到达时间`,`等待时间`,`开始点餐时间`,`结束时间`。模拟的流程图如下。




```{figure} ../_static/lecture_specific/monte-carlo/flowchart.svg
---
height: 500px
name: flow
---

```


接下来看代码实现


```{code-cell} ipython3
---
id: F4A8462A861E4F608ED0D2FB5954B80B
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np


#首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和点餐耗时：
#函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
arrivingtime = np.random.uniform(0,10,size = 20)
#.sort() 降序排列
arrivingtime.sort()
#点餐时间初始化
working = np.random.uniform(1,3,size = 20)

#到达时间、离开时间、空闲时间、等待时间初始化
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
```


```{admonition} 结论一
随机模拟下平均每人等待时间十几分钟
```

接下来改写了程序加入循环，求重复实验下每人平均等待时间


```{code-cell} ipython3
---
id: 67F939D379DF47998082277829D1C73F
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和点餐耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    arrivingtime = np.random.uniform(0,10,size=20)
    #.sort() 降序排列
    arrivingtime.sort()
    #点餐时间
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
```

```{admonition} 结论二
随机模拟重复实验下每人平均等待时间约为14.5分钟
```
要等待14.5分钟,这种情况显然必须得到改变,那我们要怎么改进呢?


如果增加一个窗口

```{code-cell} ipython3
---
id: 194906C4C05E4606BB671F8AA34FD8C1
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和点餐耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    arrivingtime = np.random.uniform(0,10,size=20)
    #.sort() 降序排列
    arrivingtime.sort()
    #点餐时间
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
```

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "ACAC10937ABE420D8D79E1D70629DBFA", "mdEditEnable": false, "jupyter": {}}


```{admonition} 结论三
增加一个窗口明显使等待时间变短
```

显然增加窗口数量是可行的，但餐厅的拥堵应该还和短时间大量的人要吃饭有关，我们来看看是不是这样

```{code-cell} ipython3
---
id: A16CB326FDC848598EB898D902B84305
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np

def forecast():
    #首先要随机生成到达时间，到达时间需要进行一下排序，方可确定排队的先后顺序和点餐耗时：
    #函数原型：  numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    
    arrivingtime = np.random.uniform(0,3,size=20)   #假设这20个人在前3分钟到达
    
    #.sort() 降序排列
    arrivingtime.sort()
    #点餐时间
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
```

```{admonition} 结论四
集中在下课后去吃饭将会使等待时间大幅增加，所以下课后不妨多等一会再去吃饭（如果不会因为去的晚没有饭的话）
```



```{tip}

**模型改进点** 

1. 学生的到达时间和点餐耗时不可能是均匀分布，应该按正态分布或者泊松分布更加合理；
1. 把用餐的总人数定在了20，可以根据实际情况取样获得用餐的总人数。
```


## 总结：蒙特卡洛模拟建模方法

**（1）构造或描述概率过程**
对于本身就具有随机性质的问题，如求解圆周率问题，主要是正确描述和模拟这个概率过程，对于本来不是随机性质的确定性问题，比如计算定积分，就必须事先构造一个人为的概率过程，它的某些参量正好是所要求问题的解。即要将不具有随机性质的问题转化为随机性质的问题。

**（2）进行随机模拟**
实现从已知概率分布抽样构造了概率模型以后，由于各种概率模型都可以看作是由各种各样的概率分布构成的，因此产生已知概率分布的随机变量（或随机向量），就成为实现蒙特卡罗方法模拟实验的基本手段，这也是蒙特卡罗方法被称为随机抽样的原因。

在计算机上，可以用物理方法产生随机数，但价格昂贵，不能重复，使用不便。另一种方法是用数学递推公式产生。这样产生的序列，与真正的随机数序列不同，所以称为伪随机数，或伪随机数序列。不过，经过多种统计检验表明，它与真正的随机数，或随机数序列具有相近的性质，因此可把它作为真正的随机数来使用。随机数是我们实现蒙特卡罗模拟的基本工具。

**（3）建立各种估计量**
一般说来，构造了概率模型并能从中抽样后，即实现模拟实验后，我们就要确定一个随机变量，作为所要求的问题的解，我们称它为无偏估计。建立各种估计量，相当于对模拟实验的结果进行考察和登记，从中得到问题的解。



## 优缺点分析

蒙特卡洛模拟的特点：随机采样得到的近似解，随着随机采样数值增多，得到正确结果的概率越大

借助计算机技术，蒙特卡洛模拟方法有两大优点
- 简单，省去了繁复的数学推导和验算过程，使普通人能够理解
- 快速，建模过程简单，确定了概率模型，后续运算完全用计算机实现

蒙特卡洛模拟方法有存在一定的缺陷
- 如果必须输入一个模式中的随机数并不像设想的那样是随机数， 而却构成一些微妙的非随机模式， 那么整个的模拟（及其预测结果）都可能是错的。




