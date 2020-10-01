# Matplotlib


本教程介绍python中进行简单数据可视化的方法，使用的工具是python中最基础的matplotlib库。



## Matplotlib图表组成



### 准备数据


数据可以手动输入，也可以通过文件进行导入，**本节内容较为简单，暂时采用手动输入的形式**，关于数据的导入方法，会在pandas教程中介绍。


为了处理数据和绘图，我们首先导入第三方包Numpy和快速绘图模块pyplot，其中科学计算包Numpy是matplotlib库的基础，也就是说，matplotlib库是建立在Numpy库基础上的Python绘图库。

import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot") #指定绘图风格为ggplot

#plt.style.available # 这里给出了其他可用的一些风格

现在我们就可以定义一些完成绘图所需的数据了，例如

x = [1,4,5,7,8,9,10]
y = [10,5,7,8,12,1,7]
plt.scatter(x,y)

x = np.linspace(0.5,3.5,10) # 在0.5到3.5之间均匀地取100个数
y = np.sin(x)           # 求出每一个数对应的正弦值
y2 = np.cos(x)
y1 = np.random.randn(10) # 在标准正态分布中随机地取100个数

plt.plot(x,y,'bo-')   # 绘制(x,y)
plt.scatter(x,y1) # 绘制(x,y1)
plt.plot(x,y2)   # 绘制(x,y)

### 绘图函数的用法

下面我们通过函数的形式来学习绘图，反过来，再用绘图的形式来强化对函数的记忆。

#### 函数`plot()` ——展现变量的变化趋势

- 函数功能：展现变量的变化趋势
- 调用方法：`plt.plot(x,y,ls='-',lw=2,label='plot figure')`
- 参数说明
    * `x`: $x$轴上的数值
    * `y`: $y$轴上的数值
    * `ls`: linestyle,折线的线条风格
    * `lw`: linewidth,折线的线条宽度
    * `label`: 标记图形内容的标签文本

x = np.linspace(0,10,51)
x

x = np.linspace(0,10,110)  # 生成从0.05 到 10 等分的1000个数据
y = np.cos(x) # 计算x的余弦值
y2 = np.sin(x) # 计算x的余弦值
plt.plot(x,y,'c',ls='-',lw=3,label='Cos')  # 线性为虚线--，线宽为6，标签为'My first line'
plt.plot(x,y2,'b',ls='-',lw=3,label='Sin')
plt.legend()   # 显示图例，不加这一句则不显示图例
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('cos.pdf')

我们也可以绘制函数图像，例如

$$
y = x^3 + 4
$$

import matplotlib.pyplot as plt 
import numpy as np
x = np.linspace(-3,3,100)
y = []
for x_i in x:
    y.append(x_i**3 + 4)
plt.plot(x,y,'b')
plt.xlabel('x')
plt.ylabel('y')

```{admonition} 课堂练习

绘制以下函数图像,并调整为你喜欢的格式。

$$
y = \sin^2(x)\cos^2(x),x\in[-3,3] 
$$
```

#### 函数`scatter()` ——寻找变量之间的关系

- 函数功能：散点图，寻找变量之间的关系
- 调用方法：`plt.scatter(x,y,c='b',label = 'scatter figure')`
- 参数说明
    * `x`: $x$轴上的数值
    * `y`: $y$轴上的数值
    * `c`: 散点图中的标记颜色
    * `label`: 标记图形内容的标签文本

x = np.linspace(0.05,10,100) # 生成从0.05 到 10 等分的100个数据
y = np.random.rand(100) # 生成100个正态分布的随机数
plt.scatter(x,y,c= 'b',label='blue') 

# y1 = np.random.rand(100) # 生成100个正态分布的随机数
# plt.scatter(x,y1,c= 'r',label='red') 
# plt.legend()

####  设置x轴的数值显示范围函数`xlim()`

- 函数功能：设置$x$轴的数值显示范围
- 调用方法：`plt.xlim(xmin,xmax)`
- 参数说明
    * `xmin`: $x$轴上的最小值
    * `xmax`: $x$轴上的最大值
    * 同样的方法可以用在`plt.ylim()`上

x = np.linspace(0.05,10,100)  # 生成从0.05 到 10 等分的100个数据
y = np.random.rand(100)# 生成100个正态分布的随机数
y1 = np.random.rand(100) # 生成100个正态分布的随机数

plt.scatter(x,y,c= 'b',label='scatter figure')  # 绘制第一组数据
plt.scatter(x,y1,c= 'r',label='scatter figure2') # 绘制第二组数据

plt.legend() # 增加图例
plt.xlim(-1,11)  #设置x轴显示范围
plt.ylim(-0.2,1.3) # 设置y轴显示范围

注意：除非特殊情况，不推荐大家自定坐标轴显示范围。

#### 函数`xlabel()` ——设置x轴的标签文本

- 函数功能：设置$x$轴的标签文本
- 调用方法：`plt.xlabel(string)`
- 参数说明
    * `string`: 标签文本内容
    * 同样的方法可以用在`plt.ylabel()`上

x = np.linspace(0.05, 10, 100)  # 生成从0.05 到 10 等分的100个数据
y = np.random.rand(100)   # 生成100个正态分布的随机数
plt.scatter(x, y, c='b', label='scatter figure')  # 绘图
plt.legend()  # 增加图例
plt.xlim(0.05, 10)  # 设置x轴显示范围
plt.ylim(0, 1)  # 设置y轴显示范围
plt.xlabel('Time') # 设置x轴标签
plt.ylabel('v1') # 设置y轴标签

#### 函数`grid() `——绘制刻度线的网格线

- 函数功能：绘制刻度线的网格线
- 调用方法：`plt.grid(linestyle = ':', color = 'r')`
- 参数说明
    * `linestyle`: 网格的线条风格
    * `color`: 网格的线条颜色

x = np.linspace(0.05,10,100)
y = np.random.rand(100)
plt.scatter(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.xlim(0.05,10)
plt.ylim(0,1)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.grid(linestyle = '-', color = 'black',alpha = 0.3)  # 增加刻度线



```{tip}

不知道一个函数的语法怎么用的时候，可以用`help()`获取，例如`help(plt.grid)`
```

#### 函数`axhline()` ——绘制平行于x轴的水平参考线

- 函数功能：绘制平行于$x$轴的水平参考线
- 调用方法：`plt.axhline(y=0.0,c='r',ls='--',lw=2)`
- 参数说明
    * `y`: 水平参考线的出发点
    * `c`: 参考线的线条颜色
    * `ls`: 参考线的线条风格
    * `lw`: 参考线的线条宽度
    * 上面的函数功能一样可以用到`axvline`上

x = np.linspace(0.05,10,100)
y = np.random.rand(100)
plt.scatter(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.xlim(0.05,10)
plt.ylim(0,1)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.axhline(y=0.5,c='r',ls='--',lw=2)  # 增加水平线
plt.axvline(x=5,c='g',ls='--',lw=2)    # 增加垂直线



plt.plot([0,10],[1,0.5],lw = 2,c = 'g')
plt.plot([0,10],[0,0.5],lw = 2,c = 'g')
# plt.plot([2,8],[0.8,0.2],lw = 2,c = 'g')

#### 函数`axvspan()` ——绘制垂直于x轴的参考区域

- 函数功能：绘制垂直于$x$轴的参考区域
- 调用方法：`plt.axvspan(xmin=1.0,xmax=2.0,facecolor='y',alpha=0.3)`
- 参数说明
    * `xmin`: 参考区域的起始位置
    * `xmax`: 参考区域的终止位置
    * `facecolor`: 参考区域的填充颜色
    * `alpha`: 参考区域的颜色透明度
    * 上面的函数功能一样可以用到`axhspan()`上

x = np.linspace(0.05,10,100)
y = np.random.rand(100)
plt.scatter(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.xlim(0.05,10)
plt.ylim(0,1)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.axhspan(ymin=0.4,ymax=0.6,facecolor='r',alpha=.1)  # 增加水平区域
plt.axvspan(xmin=4,xmax=6,facecolor='y',alpha=0.1)      # 增加垂直区域

#### 函数`annotate()` ——添加图形内容细节的指向型注释文本

- 函数功能：添加图形内容细节的指向型注释文本
- 调用方法：`plt.annotate(string,y=(x,y),xytext=(x_text,y_text),weight = 'bold',color = 'b',arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color = 'b'))`
- 参数说明
    * `string`: 图形内容的注释文本
    * `xy`: 被注释图形的内容的位置坐标
    * `xytext`: 注释文本的位置坐标
    * `weight`: 注释文本的粗细风格
    * `color`: 注释文本的字体颜色
    * `arrowprops`: 注释文本的属性字典

x = np.linspace(0.05,10,100)
y = np.sin(x)
plt.plot(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.annotate('Minimum',
            xy = (3*np.pi/2,-1), # 箭头的头部
            xytext=(3*np.pi/2-0.5,-0.5), # 箭头的尾部
            #weight = 'bold', # 粗细
            #color= 'b', # 颜色
            arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color = 'b')
            )

#### 函数`text()` ——添加图形内容细节的无指向型注释文本

- 函数功能：添加图形内容细节的指向型注释文本
- 调用方法：`plt.text(x,y,string,weight='bold',color='b')`
- 参数说明
    * `x`: 注释文本所在位置横坐标
    * `y`: 注释文本所在位置纵坐标
    * `string`: 注释文本内容
    * `weight`: 注释文本的粗细风格
    * `color`: 注释文本的字体颜色

x = np.linspace(0.05,10,100)
y = np.sin(x)
plt.plot(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.text(3.10,0.09,'sin(x)', weight = 'bold',color= 'green')

#### 函数`title()` ——添加图形内容的标题

- 函数功能：添加图形内容细节的指向型注释文本
- 调用方法：`plt.title(string)`
- 参数说明
    * `string`: 标题内容

x = np.linspace(0.05,10,100)
y = np.sin(x)
plt.plot(x,y,c= 'b',label='scatter figure')
plt.legend()
plt.text(3.10,0.09,'sin(x)', weight = 'bold',color= 'b')
plt.title('My first plot')

#### 函数`legend()` ——图例

- 函数功能：添加图形内容细节的指向型注释文本
- 调用方法：`plt.legend(loc='lower left')`
- 参数说明
    * `loc`: 图例在图中的位置

x = np.linspace(0.05,10,100)
y = np.sin(x)


y1 = np.cos(x)

plt.plot(x,y,c= 'b',label='sin')
plt.plot(x,y1,c= 'r',label='cos')
plt.text(3.10,0.09,'sin(x)', weight = 'bold',color= 'b')
plt.title('My first plot')
plt.legend()

### 绘图函数的组合应用

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm

# plt.figure(figsize=(10,5))
# define data
x = np.linspace(0.5,3.5,100)
y = np.sin(x)
y1 = np.random.randn(100)

# scatter figure
plt.scatter(x,y1,c='0.25',label='scatter figure')

# plot figure
plt.plot(x,y,ls='--',lw=2,label='plot figure')

# set x,y axis limit
plt.xlim(0,4)
plt.ylim(-3,3)

# set axis labels
plt.ylabel('y_axis')
plt.xlabel('x_axis')

# set x,y axis grid
plt.grid(ls=':',lw=2)

# add a horizontal line across the axis
plt.axhline(y=0,c='r',ls='--',lw=2)

# add a vertical span across the axis
plt.axvspan(xmin=1.0,xmax=2,facecolor = 'y',alpha = 0.3)

# set annotating info
plt.annotate('maximum',xy=(np.pi/2,1),xytext=(np.pi/2+0.15,1.5),weight='bold',color='r',
             arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color = 'r'))

plt.annotate('spines',xy=(0.75,-3),xytext=(0.35,-2.25),weight='bold',color='b',
             arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color = 'b'))

plt.annotate('',xy=(0,-2.78),xytext=(0.4,-2.32),weight='bold',color='b',
             arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color = 'b'))

plt.annotate('',xy=(3.5,-2.98),xytext=(3.6,-2.70),weight='bold',color='b',
             arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color = 'b'))

# set text info
plt.text(3.6,-2.70,'| is tickline',weight = 'bold',color = 'b')
plt.text(3.6,-2.95,'3.5 is ticklabel',weight = 'bold',color = 'b')

# set title
plt.title('structure of matplotlib')

# set legend
plt.legend()


# plt.savefig('test.pdf')

## 使用统计函数绘制简单图形

在上一部分中，我们给大家介绍了属于统计图形范畴的折线图和散点图，接下来会详细讲解一些大家比较熟悉却又经常混淆的其他几个统计图形，掌握这些统计图形可以让我们对可视化有一个更加深入地了解，并正确使用。

和前面一样，我们还是通过**函数功能、调用语法、参数说明和调用展示**四个方面为大家讲解这些函数，期望能帮助大家建立对于python数据可视化的直观认识，培养大家使用matplotlib进行进一步学习的兴趣和信心。

### 函数`bar()`——用于绘制柱状图

- 函数功能：在$x$轴上绘制定性数据的分布特征。
- 调用方法：`plt.bar(x,y)`
- 参数说明
    * `x`: 标示在$x$轴上的定性数据的分布特征
    * `y`: 每种定性数据类别的数量

import matplotlib.pyplot as plt

# some simple data
plt.figure(figsize=(10,5))

x = [1,2,3,4,5,6,7,8]
y = [3,1,4,5,8,9,7,2]

plt.bar(x,y,align='center',color='c',
                        tick_label = ['AAAAAA','BBBBBBB','CCCCC','DDDDDD','EEEEE','FFFFF','GGGGGGGG','HHHHH'])

plt.xlabel('Container No.')
plt.ylabel('Weight (kg)')
plt.grid(axis  = 'y')

### 函数`barh()`——用于绘制条形图

- 函数功能：在$y$轴上绘制定性数据的分布特征。
- 调用方法：`plt.barh(x,y)`
- 参数说明
    * `x`: 标示在$y$轴上的定性数据的分布特征
    * `y`: 每种定性数据类别的数量

import matplotlib.pyplot as plt

# some simple data
x = [1,2,3,4,5,6,7,8]
y = [3,1,4,5,8,9,7,2]

plt.barh(x,y,align='center',color='c',tick_label = ['AAAA','BBBBBBB','CCCCC','DDDDDD','EEEEE','FFFFF','GGGGGGGG','HHHHH'])

plt.ylabel('Container No.')
plt.xlabel('Weight (kg)')

```{admonition} 课堂作业

绘制疫情确诊数据柱状图，示例数据如下：
![Image Name](https://cdn.kesci.com/upload/image/q6zkf7ppgs.png?imageView2/0/w/960/h/960)
```

### 函数`hist()`——用于绘制直方图

- 函数功能：在$x$轴上绘制定量数据的分布特征。
- 调用方法：`plt.hist(x)`
- 参数说明
    * `x`: 在$x$轴上绘制箱体定量数据的输入值

import numpy as np
import matplotlib.pyplot as plt

boxWeight = np.random.randint(0,10,100)
x = boxWeight
bins = range(0,10,1)

plt.hist(x,
        bins = bins,
        color = 'c',
        histtype = 'bar',
        rwidth=0.8,
        alpha=1)

plt.xlabel('Weight of the box (kg)')
plt.ylabel('Selling Number')

##plt.savefig('bar.pdf')

### 函数`pie()`——用于绘制饼状图

- 函数功能：绘制定性数据的不同类型的百分比
- 调用方法：`plt.pie(x)`
- 参数说明
    * `x`: 定性数据的不同类型的百分比

import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3"]
soldNums = [0.05,0.45,0.15,0.35]
kinds = ['case1','case2','case3','case4']

plt.pie(soldNums,
       labels=kinds,
       autopct="%3.1f%%",
       startangle=0,
       colors=colors);

### 函数`scatter()`——用于绘制气泡图

- 函数功能：二维数据借助气泡大小展示三维数据
- 调用方法：`plt.scatter(x,y,s=size,c=color,cmap=cmap)`
- 参数说明
    * `x`: $x$轴上的数值
    * `y`: $y$轴上的数值
    * `s`: 标记的大小
    * `c`: 标记的颜色
    * `cmap`: 标记的颜色映射表

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.figure(figsize=(10,6))

a = np.random.randn(100)
b = np.random.randn(100)

plt.scatter(a,b,
           s = np.power(10*a+20*b,2),
           c = np.random.randn(100),
           cmap=mpl.cm.RdYlBu,
           marker='o')

### 函数`stem()`——用于绘制棉棒图

- 函数功能：绘制离散的有序数据
- 调用方法：`plt.stem(x,y)`
- 参数说明
    * `x`: 指定棉棒的$x$轴基线上的位置
    * `y`: 指定棉棒的长度
    * `linefmt`: 棉棒的样式
    * `markerfmt`: 棉棒末端的样式
    * `basefmt`: 基线的样式

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.5,2*np.pi,20)
y = np.random.randn(20)

plt.stem(x,y,linefmt = '-.', markerfmt = 'o', basefmt = '-')

### 函数`boxplot()`——用于绘制箱线图

- 函数功能：绘制箱线图
- 调用方法：`plt.boxplot(x)`
- 参数说明
    * `x`: 箱线图的输入数据

import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(1000)

plt.boxplot(x);
plt.xticks([1],["AlphaRM"])
plt.ylabel('Random Number')

plt.grid(linestyle = '--',alpha = 0.3)

### 函数`errorbar()`——用于绘制误差棉棒图

- 函数功能：绘制$y$轴方向或者是$x$轴方向的误差范围
- 调用方法：`plt.errorbar(x,y,yerr=a,xerr=b)`
- 参数说明
    * `x`: 数据点的水平位置
    * `y`: 数据点的垂直位置
    * `yerr`: y轴方向数据点的误差
    * `xerr`: x轴方向数据点的误差

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1,0.6,6)
y = np.exp(x)

plt.errorbar(x,y,fmt='bo:',yerr = [0.1, 0.1,0,0.2,0.1,0.15],label = 'Case1',capsize = 6)

plt.errorbar(x,y+0.5,fmt='ro:',yerr = [0.1, 0.1,0.2,0.2,0.1,0.15],label = 'Case2',capsize = 6)

plt.legend()

plt.xlabel('No of charging stations')
plt.ylabel('Dissatisfaction Index')

## 绘制进阶统计图形

### 堆积柱状图

柱状图主要是应用在定性数据的可视化场景，或者是离散数据的分布展示。例如，
- 一个本科班级的学生籍贯分布 
- 出国旅游人士的职业分布
- 下载一款APP产品的操作系统分布

关于简单的柱状图和条形图的用法，上一节已经介绍，这里我们研究更为复杂的图形。

import matplotlib.pyplot as plt

# some simple data
x = [1,2,3,4,5]
y = [6,10,4,5,1]
y1 = [2,6,3,8,5]
y2 = [1,2,3,2,1]

plt.bar(x,y,align='center',color='#66c2a5',tick_label = ['A','B','C','D','E'],label = 'ClassA')

plt.bar(x,y1,align='center',bottom = y,color='#8da0cb',tick_label = ['A','B','C','D','E'],label = 'ClassB')

plt.bar(x,y2,align='center',bottom = np.add(y,y1),color='r',tick_label = ['A','B','C','D','E'],label = 'ClassC')

plt.xlabel('Difficulty')
plt.ylabel('Number')

plt.legend()

import matplotlib.pyplot as plt

# some simple data
x = [1,2,3,4,5]
y = [6,10,4,5,1]
y1 = [2,6,3,8,5]

plt.barh(x,y,align='center',color='#66c2a5',tick_label = ['A','B','C','D','E'],label = 'ClassA')

plt.barh(x,y1,align='center',left = y,color='#8da0cb',tick_label = ['A','B','C','D','E'],label = 'ClassB')

plt.ylabel('Difficulty')
plt.xlabel('Number')

plt.legend()

### 分块柱状图

除了将多数据以堆积图的形式进行可视化展示，我们还可以使用分块图的形式。

import matplotlib.pyplot as plt

# some simple data
x = np.arange(5)
y = [6,10,4,5,1]
y1 = [2,6,3,8,5]

bar_width = 0.35
tick_label = ['A','B','C','D','E']
plt.bar(x,y,bar_width,align='center',color='#66c2a5',tick_label = tick_label,label = 'ClassA')

plt.bar(x+bar_width,y1,bar_width,align='center',color='#8da0cb',tick_label = tick_label,label = 'ClassB')

plt.xlabel('Difficulty')
plt.ylabel('Number')

plt.xticks(x+bar_width/2,tick_label)

plt.legend()

除此之外，使用`barh()`也可以达到类似的效果。

import matplotlib.pyplot as plt

# some simple data
x = np.arange(5)
y = [6,10,4,5,1]
y1 = [2,6,3,8,5]

bar_width = 0.35
tick_label = ['A','B','C','D','E']
plt.barh(x,y,bar_width,align='center',color='#66c2a5',tick_label = tick_label,label = 'ClassA')

plt.barh(x+bar_width,y1,bar_width,align='center',color='#8da0cb',tick_label = tick_label,label = 'ClassB')

plt.ylabel('Difficulty')
plt.xlabel('Number')

plt.xticks(x+bar_width/2,tick_label)

plt.legend()

```{admonition} 课堂练习

绘制确诊数据、疑似数据、死亡数据和治愈数据的柱状图。

![Image Name](https://cdn.kesci.com/upload/image/q6zkf7ppgs.png?imageView2/0/w/600/h/600)

```


### 堆积折线图

堆积柱状图的本质是将若干折线图放在同一个坐标轴上，以每条折线下部和下方折线作为填充边界，用一种颜色代表此折线的数值区域。

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1,6,1)
y = [0,4,3,5,6]
y1 = [1,3,4,2,7]
y2 = [3,4,1,6,5]

labels = ["BluePlanet","BrownPlanet","GreenPlanet"]
colors = ["#8da0cb","#fc8d62","#66c2a5"]

plt.stackplot(x,y,y1,y2,labels = labels, colors = colors)

plt.legend(loc = 'upper left')

### 间断条形图

间断条形图是在条形图的基础上进行绘制的，主要用来可视化定性数据的相同指标在时间维度上的指标值。该方法是通过`broken_barh()`实现的

import matplotlib.pyplot as plt
import numpy as np

plt.broken_barh([(30,100),(180,50),(260,70)],(20,8),facecolor = '#1f78b4')

plt.broken_barh([(60,90),(190,20),(230,30),(280,60)],(10,8),
                facecolor = ('#7fc97f','#beaed4','#fdc086','#ffff99'))

plt.xlim(0,360)
plt.ylim(5,35)
plt.xlabel('Time')

plt.xticks(np.arange(0,361,60))
plt.yticks([15,25],['A','B'])
# plt.grid()

### 内嵌环形饼图

我们先来回忆一下饼状图的绘制方法。

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))

colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3"]
soldNums = [0.05,0.45,0.15,0.35]
kinds = ['case1','case2','case3','case4']

explode = (0,0,0,0.35)

plt.pie(soldNums,
        explode = explode,
       labels=kinds,
       autopct="%3.1f%%",
       startangle=60,
       colors=colors);

饼图不仅可以展示单一数据的分布情况，还可以通过内嵌式的环形饼图实现多个数据集之间的对比。

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6,6))
elements = ["Flour","Sugar","Cream","Strawberry","Nuts"]

weight1 = [40,15,20,10,15]
weight2 = [30,25,15,20,10]

colormaplist = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]
outer_color = colormaplist
inner_color = colormaplist

wedges1,texts1,autotexts1 = plt.pie(weight1,
                                   autopct="%3.1f%%",
                                   radius=1,
                                   pctdistance=0.85,
                                   colors=outer_color,
                                   textprops=dict(color= "w"),
                                   wedgeprops=dict(width=0.3, edgecolor = 'w'))

wedges1,texts1,autotexts1 = plt.pie(weight2,
                                   autopct="%3.1f%%",
                                   radius=0.75,
                                   pctdistance=0.75,
                                   colors=outer_color,
                                   textprops=dict(color= "w"),
                                   wedgeprops=dict(width=0.3, edgecolor = 'w'))
plt.legend(wedges1,elements,fontsize = 12, loc = 'center right',bbox_to_anchor = (1,0,0.3,1))

### 箱线图

箱线图是由一个箱体和一对箱须所组成的统计图形。箱体是由第一四分位数、中位数（第二四分位数）、和第三四分位数组成的。在箱须末端之外的数值可以理解为离群值，因此，箱须是对一组数据范围的大致直观描述。

箱线图主要应用于一系列测量或者观测数据的比较场景中，例如学校间或班级间的测试成绩比较，球队中的队员体能对比，产品优化前后的测试比较以及同类产品的各项性能之间的比较等等。箱线图的应用非常广泛，实现也很简单。

import matplotlib.pyplot as plt
import numpy as np

testA = np.random.randn(5000)
testB = np.random.randn(5000)

testlist = [testA,testB]
labels = ["AlphaRM","BetaRM"]

whis = 2.5 # 四分位间距的倍数
width = 0.35

plt.boxplot(testlist,
           whis= whis,
           widths=width,
           sym='o',   # 离群值的标记样式
           labels=labels,
           notch=True,
           patch_artist=False); # 是否给箱体添加颜色
plt.ylabel('Random Number')

plt.grid(linestyle = '--',alpha = 0.3)

<font color=green size=3>  延伸阅读：箱体、箱须、离群值的含义和计算方法 </font>


- [箱线图](https://baike.baidu.com/item/%E7%AE%B1%E5%BD%A2%E5%9B%BE/10671164?fromtitle=%E7%AE%B1%E7%BA%BF%E5%9B%BE&fromid=10101649)
- [箱线图，维基百科 ](https://zh.wikipedia.org/zh-hans/%E7%AE%B1%E5%BD%A2%E5%9C%96)


## 完善统计图形

### 添加图例和标题

在绘图区域可能有多个图形，而这些图形如果不加说明，观察者则很难辨识出这些图形的主要内容。因此，我们需要对这些图形添加标签进行说明，这些标签就是图例。同样，一个言简意赅的标题同样能够帮助观察者了解图形的绘制内容。

#### 图例和标题的设置方法

图例和标签的设置主要使用`legend()`和`title()`方法

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.linspace(-2*np.pi,2*np.pi,200)
y = np.sin(x)
y1 = np.cos(x)

plt.plot(x,y,label = "$\sin(x)$")
plt.plot(x,y1,label = "$\cos(x)$")
plt.xlabel(r'x')
plt.legend()

### 图形大小、字体和字号的设置

有的时候我们想调整图形的字号大小，方便观察。

plt.figure(figsize=(15,6))  # 设置图像的大小
x = np.linspace(-2*np.pi,2*np.pi,200)
y = np.sin(x)
y1 = np.cos(x)
plt.plot(x,y,label = 'sin(x)')
plt.plot(x,y1,label = 'cos(x)')
plt.legend(fontsize = 20)  # 设置图例的大小
plt.title('$f(x) = sin(x)$',fontsize = 20)  # 设置标题的大小
plt.tick_params(labelsize=20)  # 设置刻度的字体大小
plt.xlabel(r'$x$',fontsize = 20)  # 设置横轴标签的字体大小
plt.ylabel(r'$f(x)$',fontsize = 20)  # 设置纵轴标签的字体大小

### 多个子图的绘制

子图的本质是将绘图区划分为网格，在纵横交错的并列网格中，添加绘图坐标轴。实现了一张画图绘制多张图片。

#### 函数`subplot()`：绘制网格区域中的几何形状相同的子区布局

import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi,2*np.pi,200)
plt.figure(figsize=(10,4))
y = np.sin(x)
y1 = np.cos(x)


plt.subplot(2,3,1)
plt.plot(x,y,label = "$\sin(x)$")



plt.subplot(2,3,2)
plt.plot(x,y1,label = "$\cos(x)$")

plt.subplot(2,3,3)
plt.plot(x,y,label = "$\sin(x)$")
plt.subplot(2,3,6)
plt.plot(x,y1,label = "$\cos(x)$")
#plt.savefig('test1.pdf')

## 实用案例：多个统计图形的组合展示

这里介绍一个一种可行的写法，不过新手也可以单独绘制每一张图，然后将其手动拼合在一起。

import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(2,3,figsize = (15,10))

# subplot(2,3,1)
colors = ['#8dd3c7','#ffffb3','#bebada']
ax[0,0].bar([1,2,3],[0.6,0.8,0.2],color = colors, width = 0.5, hatch = '///',align = 'center')

# subplot(2,3,2)
x = np.linspace(-2*np.pi,2*np.pi,200)
y = np.sin(x)
y1 = np.cos(x)
ax[0,1].plot(x,y,'r',label = "$\sin(x)$")
ax[0,1].plot(x,y1,'b',label = "$\cos(x)$")
ax[0,1].legend()  # 设置图例的大小
plt.xlabel('$x$')  # 设置横轴标签的字体大小
plt.ylabel('$f(x)$')  # 设置纵轴标签的字体大小

# subplot(2,3,3)
x = [1,2,3,4,5,6,7,8]
y = [3,1,4,5,8,9,7,2]

ax[0,2].barh(x,y,align='center',color='c',tick_label = ['A','B','C','D','E','F','G','H'])

plt.ylabel('Container No.')
plt.xlabel('Weight (kg)')

# subplot(2,3,4)
import matplotlib as mpl
a = np.random.randn(100)
b = np.random.randn(100)

ax[1,0].scatter(a,b,
           s=np.power(10*a+20*b,2),
           c = np.random.randn(100),
           cmap=mpl.cm.RdYlBu,
           marker='o')

# subplot(2,3,5)
x = np.linspace(0.5,2*np.pi,20)
y = np.random.randn(20)

ax[1,1].stem(x,y,linefmt = '-.', markerfmt = 'o', basefmt = '-')



# subplot(2,3,6)
elements = ["Flour","Sugar","Cream","Strawberry","Nuts"]

weight1 = [40,15,20,10,15]
weight2 = [30,25,15,20,10]

colormaplist = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]
outer_color = colormaplist
inner_color = colormaplist

wedges1,texts1,autotexts1 = ax[1,2].pie(weight1,
                                   autopct="%3.1f%%",
                                   radius=1,
                                   pctdistance=0.85,
                                   colors=outer_color,
                                   textprops=dict(color= "w"),
                                   wedgeprops=dict(width=0.3, edgecolor = 'w'))

wedges1,texts1,autotextdds1 = ax[1,2].pie(weight2,
                                   autopct="%3.1f%%",
                                   radius=0.75,
                                   pctdistance=0.75,
                                   colors=outer_color,
                                   textprops=dict(color= "w"),
                                   wedgeprops=dict(width=0.3, edgecolor = 'w'))
plt.legend(wedges1,elements,fontsize = 12, loc = 'center right',bbox_to_anchor = (1,0,0.3,1))
# plt.savefig('group.pdf')

## 课后作业


```{admonition} 课后作业
1.绘制函数图像
 
$$
f(x) = \sin^2(x−2)e^{−x^2} ,x \in [0,2]
$$

加上适当的标题，坐标轴说明。

2.按照如下要求，以2*2的比例绘制成一个大图（注意给每个图片加上$x$轴，$y$轴，和标题）。
- $f(x) = \sin(x)$的曲线图
- 课堂练习中绘制的柱状图
- 课堂练习中绘制的饼状图
- 任意一个课程学到的，你想重现的图形

```