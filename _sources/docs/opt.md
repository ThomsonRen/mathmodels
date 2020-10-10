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

# 优化算法


回忆优化算法三要素：

- **变量**（Decision Variable）
- **约束条件**（Constraints）
- **目标函数**（Objective function）

采用最优化算法解决实际问题主要分为下列两步：

1. 建立数学模型。对可行方案进行编码（就是确定变量），约束条件以及目标函数的构造。
2. 指定最优值的搜索策略。有精确算法、穷举、随机和启发式搜索方法等。



例如，对于一组可以用列向量表示的变量

$$
X = \left[ x_1,x_2,x_3,\cdots,x_n  \right]^T
$$

我们的目的是寻找目标函数的最大或最小值

$$
\max f(X)  \quad \text{or} \quad  \min f(X)
$$

此时通常需要满足一定的约束条件，约束条件可分为等式约束和不等式约束：


$$
s.t. \begin{cases}
g_{i}(X) \leq 0   \quad (i = 1,2, \ldots, n)\\
h_{j}(X)=0 \quad (j=1,2, \ldots, m)\end{cases}
$$


此外，根据自变量的情况，也可以将最优化问题分为**函数优化问题**和**组合优化问题**两大类，其中函数优化的对象是一定区间的连续变量，而组合优化的对象则是解空间中的离散状态。其中典型的组合优化问题有。

- 旅行商(Traveling salesman problem,TSP)问题
- 加工调度问题(Scheduling problem,如Flow-shop,Job-shop)
- 0-1背包问题(Knapsack problem)
- 装箱问题(Bin packing problem)
- 图着色问题(Graph coloring problem)
- 聚类问题(Clustering problem)等


## TSP问题

已知敌方100个目标的经度、纬度如下表所示。

![Image Name](https://cdn.kesci.com/upload/image/q3w4cm6jo6.png?imageView2/0/w/960/h/960)

![Image Name](https://cdn.kesci.com/upload/image/q3w4ctq050.png?imageView2/0/w/960/h/960)


我方有一个基地，经度和纬度为$(70,40)$。假设我方飞机的速度为$1000公里/小时$。我方派一架飞机从基地出发，侦察完敌方所有目标，再返回原来的基地。在敌方每一目标点的侦察时间不计，求该架飞机所花费的时间（假设我方飞机巡航时间可以充分长）。我们依次给基地编号为$1$，敌方目标依次编号为$2,3,\cdots,101$，最后我方基地再重复编号为 $102$。

设距离矩阵为

$$
D=\left(d_{i j}\right)_{102 \times 102}
$$

其中，$d_{ij}$表示$i,j$两点之间的距离$(i, j=1,2, \cdots, 102)$。

则问题是求一个从点1 出发，走遍所有中间点，到达点102 的一个最短路径。


将上述问题抽象为即为**旅行商问题 ( TSP,Traveling Salesman Problem )** ：有N个城市，要求从其中某个城市出发，唯一遍历所有城市，再回到出发的城市，求最短的路线。

旅行商问题属于所谓的**NP完全问题**（是世界七大数学难题之一。NP的英文全称是Non-deterministic Polynomial的问题，即多项式复杂程度的非确定性问题），**精确的解决TSP只能通过穷举所有的路径组合**，其时间复杂度是$O(N!)$ 。

也就是说，针对以上问题，如果我们使用穷举的方法，需要计算$A_{102}^{102} = 102!$种情况，然后选择一个最好的解，但这种方法的计算量显然是无法接受的。我们称类似的问题为NP-hard 组合优化问题。

针对这类问题，我们可以考虑采用现代优化算法进行求解。


## 现代优化算法的分类


现代优化算法是 80 年代初兴起的启发式算法。这些算法包括

- 禁忌搜索（tabu search）
- 模拟退火（simulated annealing）
- 遗传算法（genetic algorithms）
- 人工神经网络（neural networks）等

它们主要用于解决大量的实际应用问题。目前，这些算法在理论和实际应用方面得到了较大的发展。现代启发式算法在优化机制方面存在一定的差异,但在优化流程上却具有较大的相似性,均是一种**“邻域搜索”**结构。算法都是从一个(一组)初始解出发,在算法的关键参数的控制下通过邻域函数产生若干邻域解,按接受准则(确定性、概率性或混沌方式)更新当前状态,而后按关键参数修改准则调整关键参数。如此重复上述搜索步骤直到满足算法的收敛准则,最终得到问题的优化结果。

启发式算法包含的算法很多，例如解决复杂优化问题的蚁群算法。有些启发式算法是根据实际问题而产生的，如解空间分解、解空间的限制等；另一类算法是集成算法，这些算法是诸多启发式算法的合成。


我们通常把启发式算法分为以下四类：

- 基于随机选择
	- 通过随机选择之后进行筛选
	
- 基于贪心算法
	- 通过每一步尝试最优策略
	
- 基于元启发式算法
	- 通过已有的较优解迭代
	- 基于禁忌搜索、基于遗传算法、基于遗传编程、基于蚁群算法

- 基于学习
	- 通过构造参数学习函数或特征匹配
	- 深度学习、机器学习


今天，我们将会学习启发式算法中的两个最经典的算法：**遗传算法和模拟退火算法**。


## 遗传算法


遗传算法（Genetic Algorithm）是模拟达尔文生物进化论的自然选择和遗传学机理的生物进化过程的计算模型，是一种通过模拟自然进化过程搜索最优解的方法。遗传算法是在20世纪六七十年代由美国密歇根大学的 Holland教授创立。

遗传算法是一种仿生算法，最主要的思想是**物竞天择，适者生存**。这个算法很好的模拟了生物的进化过程，保留好的物种，同样一个物种中的佼佼者才会幸运的存活下来。转换到数学问题中，这个思想就可以很好的转化为优化问题，在求解方程组的时候，好的解视为好的物种被保留，坏的解视为坏的物种而淘汰，设置好进化次数以后开始迭代，记录下这些解里面最好的那个，就是要求解的方程组的解。

遗传算法是从代表问题可能潜在的解集的一个**种群**（population）开始的，而一个种群则由经过**基因**（gene）编码的一定数目的**个体**(individual)组成。每个个体实际上是**染色体**(chromosome)带有特征的实体。染色体作为遗传物质的主要载体，即多个基因的集合，其内部表现（即基因型）是某种基因组合，它决定了个体的形状的外部表现，如黑头发的特征是由染色体中控制这一特征的某种基因组合决定的。

+++ {"id": "E5F1B461D46948D5807F7FAE980D7AA9", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

***

+++ {"id": "5D1B9F9438AA4EAAB879BBB2E619E710", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 模型引入:函数寻优问题

例子：已知一元函数：
$$
f(x)=2 x \sin ( x)+\cos(x) ,x \in[0,10]
$$

求该指定区间内的最大值。我们先来看下函数的图像。


```{code-cell} ipython3
---
id: 702678C4D8554AB7858E721BB50AFEE0
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
x = [0.01*i for i in range(0,1000)]
y = [2 *i * np.sin( i) + np.cos(i) for i in x]
plt.plot(x,y)
plt.legend()
```

可以看到这个函数有许多局部极大值点，使用传统的搜索算法很容易落入“局部最优”。



为了帮助大家理解遗传算法的思想，我们引入一个比喻：**“袋鼠跳”问题**


可以把函数优化的过程看作是一个在多元函数里面求最优解的过程。可以这样想象，在多维曲面里面有数不清的“山峰”，而这些山峰所对应的就是局部最优解。而其中也会有一个“山峰”的海拔最高的，那么这个就是全局最优解。而优化算法的任务就是尽量爬到最高峰，而不是陷落在一些小山峰。


![Image Name](https://cdn.kesci.com/upload/image/q0ls5rey9d.png?imageView2/0/w/960/h/960)

> 注意：以下求解过程均假设求解最大值。(思考，如果需要求最小值怎么办？)



## 数学原理/实现过程


这里我们先回顾几个生物学概念

**基因型(genotype)**：性状染色体的内部表现；

**表现型(phenotype)**：染色体决定的性状的外部表现，或者说，根据基因型形成的个体的外部表现；

**进化(evolution)**：种群逐渐适应生存环境，品质不断得到改良。生物的进化是以种群的形式进行的。

**适应度(fitness)**：度量某个物种对于生存环境的适应程度。

**选择(selection)**：以一定的概率从种群中选择若干个个体。一般，选择过程是一种基于适应度的优胜劣汰的过程。

**复制(reproduction)**：细胞分裂时，遗传物质DNA通过复制而转移到新产生的细胞中，新细胞就继承了旧细胞的基因。

**交叉(crossover)**：两个染色体的某一相同位置处DNA被切断，前后两串分别交叉组合形成两个新的染色体。也称基因重组或杂交；

**变异(mutation)**：复制时可能（很小的概率）产生某些复制差错，变异产生新的染色体，表现出新的性状。

**编码(coding)**：DNA中遗传信息在一个长链上按一定的模式排列。遗传编码可看作从表现型到基因型的映射。

**解码(decoding)**：基因型到表现型的映射。

**个体（individual）**：指染色体带有特征的实体；

**种群（population）**：个体的集合，该集合内个体数称为种群的大小。



既然我们把函数曲线理解成一个一个山峰和山谷组成的山脉。那么我们可以设想所得到的**每一个解就是一只袋鼠**，我们希望它们不断的向着更高处跳去，直到跳到最高的山峰。所以求**最大值**的过程就转化成一个“**袋鼠跳**”的过程。

我们假设，在这个故事中，有很多袋鼠，它们被随机分配到喜玛拉雅山脉的任意地方。每过几年，就在一些**海拔高度较低的地方射杀一些袋鼠**，那些存活下来的袋鼠在它们所处的地方生儿育女。**结果**就这样经过许多年，这些袋鼠们竟然都不自觉地聚拢到了一个个的山峰上，可是在所有的袋鼠中，只有聚拢到珠穆朗玛峰的袋鼠被带回了美丽的澳洲（最优解）。



遗传算法的实现过程实际上就像自然界的进化过程那样。
- 首先寻找一种对问题潜在解进行**“数字化”编码**的方案。（建立表现型和基因型的映射关系）
- 然后用**随机数初始化**一个种群（那么第一批袋鼠就被随意地分散在山脉上），种群里面的个体就是这些数字化的编码。
- 接下来，通过适当的**解码**过程之后（得到袋鼠的位置坐标），用**适应性**函数对每一个基因个体作一次适应度评估（袋鼠爬得越高，越是受我们的喜爱，所以适应度相应越高）。
- 用选择函数按照某种规定**择优选择**（我们要每隔一段时间，在山上射杀一些所在海拔较低的袋鼠，以保证袋鼠总体数目持平）。
- 让**个体基因变异**（让袋鼠随机地跳一跳）。然后**产生子代**（希望存活下来的袋鼠是多产的，并在那里生儿育女）。

遗传算法并不保证你能获得问题的最优解，但是使用遗传算法的最大优点在于你不必去了解和操心如何去“找”最优解。（你不必去指导袋鼠向那边跳，跳多远。）而只要简单的“否定”一些表现不好的个体就行了。



> 把那些总是爱走下坡路的袋鼠射杀，这就是遗传算法的精粹！

+++ {"id": "8B2A1744EA7A4F2C8069DE216B4342EE", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 编制袋鼠的染色体----基因的编码方式

+++ {"id": "79208BF03CBF44F1B3D356009D62389D", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 二进制编码法

+++ {"id": "1B7DE6CD7B8E4622A14EC2CAB670AF2B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

受到人类染色体结构的启发，我们可以设想一下，假设目前只有“0”，“1”两种碱基，我们也用一条链条把他们有序的串连在一起，因为每一个单位都能表现出 1 bit的信息量，所以一条足够长的染色体就能为我们勾勒出一个个体的所有特征。这就是二进制编码法，染色体大致如下：

$$
010010011011011110111110
$$

+++ {"id": "630A3CFB62D7489AA4732929CC94B566", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 浮点数编码

+++ {"id": "2F2382E0AB86404C80807CD441CCFCAB", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

上面的编码方式虽然简单直观，但明显地，当个体特征比较复杂的时候，需要大量的编码才能精确地描述，相应的解码过程（类似于生物学中的DNA翻译过程，就是把基因型映射到表现型的过程）将过分繁复，为改善遗传算法的计算复杂性、提高运算效率，提出了浮点数编码。染色体大致如下：
$$
1.2 –3.3 – 2.0 –5.4 – 2.7 – 4.3
$$

+++ {"id": "DC0EC7DC0AAC491F8A3CFD7F0FACF797", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 只编码主要特征

+++ {"id": "EB1F8C3D73374F22A211C15A6AE1B528", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

那么我们如何利用这两种编码方式来为袋鼠的染色体编码呢？因为编码的目的是建立表现型到基因型的映射关系，而表现型一般就被理解为个体的特征。比如人的基因型是46条染色体所描述的却能解码成一个眼，耳，口，鼻等特征各不相同的活生生的人。所以我们要想为“袋鼠”的染色体编码，我们必须先来考虑“袋鼠”的“个体特征”是什么。也许有的人会说，袋鼠的特征很多，比如性别，身长，体重，也许它喜欢吃什么也能算作其中一个特征。但具体在解决这个问题的情况下，我们应该进一步思考：无论这只袋鼠是长短，肥瘦，黑白，**只要它在低海拔就会被射杀**，同时也没有规定身长的袋鼠能跳得远一些，身短的袋鼠跳得近一些。当然它爱吃什么就更不相关了。

**我们由始至终都只关心一件事情：袋鼠在哪里**。

因为只要我们知道袋鼠在那里，我们就能做两件必须去做的事情：

1、通过查阅喜玛拉雅山脉的地图来得知袋鼠所在的海拔高度（通过自变量求适应函数的值）以判断我们有没必要把它射杀。

2、知道袋鼠跳一跳（交叉和变异）后去到哪个新位置。

+++ {"id": "C518ECCD7BF94BC6831DD9AC3C6A5964", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

> Tips:必须把具体问题抽象成数学模型，突出主要矛盾，舍弃次要矛盾。只有这样才能简洁而有效的解决问题。

+++ {"id": "0592A3230DC24323858EA55542C86F53", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

## 物竞天择－－适应性评分与及选择函数

+++ {"id": "271FC0DD88864C7682336145B60A5D6B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 物竞――适应度函数（fitness function）

+++ {"id": "334AE0A39C2248DCAACC97317E58E8EA", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

自然界生物竞争过程往往包含两个方面：生物相互间的搏斗与及生物与客观环境的搏斗过程。但在我们这个实例里面，你可以想象到，袋鼠相互之间是非常友好的，它们并不需要互相搏斗以争取生存的权利。它们的生死存亡更多是取决于你的判断。因为你要衡量哪只袋鼠该杀，哪只袋鼠不该杀，所以你必须制定一个衡量的标准。而对于这个问题，这个衡量的标准比较容易制定：

**袋鼠所在的海拔高度。（因为你单纯地希望袋鼠爬得越高越好）**


所以我们直接用袋鼠的海拔高度作为它们的适应性评分。即适应度函数直接返回函数值就行了。


一般在我们的优化问题中，目标函数就是适应度函数。

+++ {"id": "A7C6B3715DED4566B0FF323DD46F0AC0", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 天择――选择函数（selection）

+++ {"id": "9A8CF8EA3FAA4CC8808867571200684F", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

自然界中，越适应的个体就越有可能繁殖后代。但是也不能说适应度越高的就肯定后代越多，只能是从概率上来说更多。（毕竟有些所处海拔高度较低的袋鼠很幸运，逃过了你的眼睛。）那么我们怎么来建立这种概率关系呢？下面我们介绍一种常用的选择方法――**轮盘赌（Roulette Wheel Selection）选择法**。

+++ {"id": "7711FCD37ABF4F1A830FB76A11E2D601", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

比如我们有5条染色体，他们所对应的适应度评分分别为：$5，7，10，13，15$

所以累计总适应度为：

$$
F=\sum_{i=1}^{n} f_{i}=5+7+10+13+15=50
$$

所以各个个体被选中的概率分别为：

$$
\begin{array}{l}{P_{1}=\frac{f_{1}}{F} \times 100 \%=\frac{5}{50} \times 100 \%=10 \%} \\ {P_{2}=\frac{f_{2}}{F} \times 100 \%=\frac{7}{50} \times 100 \%=14 \%} \\ {P_{3}=\frac{f_{3}}{F} \times 100 \%=\frac{10}{50} \times 100 \%=20 \%} \\ {P_{4}=\frac{f_{4}}{F} \times 100 \%=\frac{13}{50} \times 100 \%=26 \%} \\ {P_{5}=\frac{f_{5}}{F} \times 100 \%=\frac{15}{50} \times 100 \%=30 \%}\end{array}
$$


![Image Name](https://cdn.kesci.com/upload/image/q0ls7hac6t.png?imageView2/0/w/960/h/960)

+++ {"id": "CF1D028EF9DD41199F7863CF4402506F", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

适应度评分越高的个体被选中的概率越大，也就是繁殖后代的概率越大。

当然，我们也可以使用其他的选择方式，比如不引入概率选择，只保留那些性状最好的个体。

+++ {"id": "BB6E81DB0632487E8717E13749AFBA43", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

## 遗传变异――基因重组（交叉）与基因突变

+++ {"id": "C16B24425D27425C85EE39EFF9697083", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**基因重组与基因突变是使得子代不同于父代的根本原因**。对于这两种遗传操作，二进制编码和浮点型编码在处理上有很大的差异，其中二进制编码的遗传操作过程，比较类似于自然界里面的过程，下面将分开讲述。

+++ {"id": "EC4520E777F643B5879B594274E99D67", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

### 基因重组/交叉(recombination/crossover)

+++ {"id": "F6822A999B444CC1814C7591385AD4E2", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

先来看二进制编码的情况。  二进制编码的基因交换过程非常类似高中生物中所讲的同源染色体的联会过程――随机把其中几个位于同一位置的编码进行交换，产生新的个体。

+++ {"id": "75EBE7015BE34A738B12AF33AEC41203", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}


![Image Name](https://cdn.kesci.com/upload/image/q0ls87ho9o.png?imageView2/0/w/960/h/960)

+++ {"id": "22AF534CD257466A82761F9846D9EFD3", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

再来看浮点数编码的情况。如果一条基因中含有多个浮点数编码，那么也可以用跟上面类似的方法进行基因交叉，不同的是进行交叉的基本单位不是二进制码，而是浮点数。而如果对于单个浮点数的基因交叉，就有其它不同的重组方式了，比如**中间重组：随机产生就能得到介于父代基因编码值和母代基因编码值之间的值作为子代基因编码的值**。比如5.5和6交叉，产生5.7，5.6。

+++ {"id": "4A7FC8FB066E4529B02FAF51466167DB", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

> 注意，子代不一定优于父代，**只有经过自然的选择后，才会出现子代优于父代的倾向**。

+++ {"id": "81332D83158249528D5B1CC29763E7C6", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 基因突变(Mutation)

+++ {"id": "3CB453C0FFD3423289732874304E9DD8", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**先来看二进制编码的情况。** 基因突变过程：基因突变是染色体的**某一个位点上基因的改变**。基因突变使一个基因变成它的等位基因，并且通常会引起一定的表现型变化。正如上面所说，二进制编码的遗传操作过程和生物学中的过程非常相类似，基因串上的“ 0”或“ 1”有一定几率变成与之相反的“ 1”或“ 0”。例如下面这串二进制编码：
$$
101101001011001
$$

经过基因突变后，可能变成以下这串新的编码：
$$
001101011011001
$$

+++ {"id": "81283AB9FE1A41BC859E63C2838E28F0", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**再来看浮点数编码的情况。** 浮点型编码的基因突变过程一般是对原来的浮点数**增加或者减少一个小随机数**。比如原来的浮点数串如下：
$$
1.2,3.4,5.1, 6.0, 4.5
$$

变异后，可能得到如下的浮点数串：
$$
1.3,3.1,4.9, 6.3, 4.4
$$

  当然，这个小随机数也有大小之分，我们一般管它叫“**步长**”。（想想“袋鼠跳”问题，袋鼠跳的长短就是这个步长。）一般来说**步长越大，开始时进化的速度会比较快，但是后来比较难收敛到精确的点上。**而小步长却能较精确的收敛到一个点上。

+++ {"id": "E3AFE1EF18CC44889C273C93A124A8B1", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

> 很多时候为了加快遗传算法的进化速度，而又能保证后期能够比较精确地收敛到最优解，会采取动态改变步长的方法。

+++ {"id": "87CB41A83A6D4E2E9E2B11386025CF94", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

***

+++ {"id": "4B9A65BB86AC44FAABB6BCC3701B3C2B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

至此，我们了解了遗传算法的各个步骤，流程图如下：

+++ {"id": "79EAF02530574C449EB96773EF214853", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}


![Image Name](https://cdn.kesci.com/upload/image/q0ls8xaygm.jpg?imageView2/0/w/960/h/960)

+++ {"id": "A7595A06FF114F20BB81D850DDD8E97F", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

***

+++ {"id": "61E11F1A2F65494886C82B569402EE08", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

# 遗传算法案例代码求解

+++ {"id": "8DB2D0950C3B41C18B3B128F52693CA6", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

现在我们用 Python 来实现遗传算法

+++ {"id": "F5FCC7ED079643418994C52275947FC2", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

1.种群初始化


(1)随机生成若干个数的二进制染色体。

```{code-cell} ipython3
---
id: BB80B7B6D6B24C609456B5FC18A19174
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
# -*-coding:utf-8 -*-
#目标求解2*x*sin(x)+cos(x)最大值
import random
import math
import matplotlib.pyplot as plt
#初始化生成chromosome_length大小的population_size个个体的二进制基因型种群
def species_origin(population_size,chromosome_length):
    population=[[]]
    #二维列表，包含染色体和基因
    for i in range(population_size):
        temporary=[]
        #染色体暂存器
        for j in range(chromosome_length):
            temporary.append(random.randint(0,1))
            #随机产生一个染色体,由二进制数组成
        population.append(temporary)
            #将染色体添加到种群中
    return population[1:]
            # 将种群返回，种群是个二维数组，个体和染色体两维
```

+++ {"id": "4D18434520A94A18A3FE293016441781", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

(2)编码

将二进制的染色体基因型编码成十进制的表现型。

```{code-cell} ipython3
---
id: 166137C0314245FC8C1629C8032D9BBC
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
#从二进制到十进制
 #input:种群,染色体长度
def translation(population,chromosome_length):
    temporary=[]
    for i in range(len(population)):
        total=0
        for j in range(chromosome_length):
            total+=population[i][j]*(math.pow(2,j))
            #从第一个基因开始，每位对2求幂，再求和
            # 如：0101 转成十进制为：1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 0 * 2^3 = 1 + 0 + 4 + 0 = 5
        temporary.append(total)
        #一个染色体编码完成，由一个二进制数编码为一个十进制数
    return temporary
   # 返回种群中所有个体编码完成后的十进制数
```

+++ {"id": "881C81A3B09A4C15BC80B84045A2DAF5", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

2.适应度计算

个体适应度与其对应的个体表现型$x$的目标函数值相关联，$x$越接近于目标函数的最优点，其适应度越大，从而其存活的概率越大。反之适应度越小，存活概率越小。这就引出一个问题关于适应度函数的选择，本例中，函数值总取非负值，**以函数最大值为优化目标，故直接将目标函数作为适应度函数**。这里我们直接将目标函数$2x\sin(x)+\cos(x)$作为个体适应度。如果，你想优化的是多元函数的话，需要将个体中基因型的每个变量提取出来，分别带入目标函数。比如说：我们想求$x_1+\ln(x_2)$的最大值。基因编码为4位编码，其中前两位是$x_1$，后两位是$x_2$。那么我们在求适应度的时候，需要将这两个值分别带入$f(x_1)=x$,$f(x_2)=\ln x$。再对$f(x_1)$和$f(x_2)$求和得到此个体的适应度。在本篇中，虽然染色体长度为10,但是实际上只有一个变量。

```{code-cell} ipython3
---
id: 85C57889BDE148FDB310276CA50DC611
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
# 目标函数相当于环境 对染色体进行筛选，这里是2*sin(x)+cos(x)
def function(population,chromosome_length,max_value):
    temporary=[]
    function1=[]
    temporary=translation(population,chromosome_length)
    # 暂存种群中的所有的染色体(十进制)
    for i in range(len(temporary)):
        x=temporary[i]*max_value/(math.pow(2,chromosome_length)-1)
        #一个基因代表一个决策变量，其算法是先转化成十进制，然后再除以2的基因个数次方减1(固定值)。
        #function1.append(2*math.sin(x)+math.cos(x))
        function1.append(2*x*math.sin( x)+math.cos(x))
        #这里将2*sin(x)+cos(x)作为目标函数，也是适应度函数
    return function1
```

+++ {"id": "8A68C625C39549E583973CD2D8B76CBB", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

3.选择操作

(1).只保留非负值的适应度/函数值(不小于0)——避免轮盘赌出现负的概率

```{code-cell} ipython3
---
id: E86752FB4ADB4C0988FF83DED396EC93
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
def fitness(function1):
    fitness1=[]
    min_fitness=mf=0
    for i in range(len(function1)):
        if(function1[i]+mf>0):
            temporary=mf+function1[i]
        else:
            temporary=0.0
        # 如果适应度小于0,则定为0
        fitness1.append(temporary)
        #将适应度添加到列表中
    return fitness1
```

+++ {"id": "8330FAD0DBBB4DB1A7E12C5149C4E3D8", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

(2).首先计算出所有个体的适应度总和$\sum f_i$

```{code-cell} ipython3
---
id: A917643F84B44604A327B989B435E099
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
#计算适应度和
def sum(fitness1):
    total=0
    for i in range(len(fitness1)):
        total+=fitness1[i]
    return total
 
#计算适应度斐波纳挈列表，这里是为了求出累积的适应度
def cumsum(fitness1):
    for i in range(len(fitness1)-2,-1,-1):
        # range(start,stop,[step])
        # 倒计数
        total=0
        j=0
        while(j<=i):
            total+=fitness1[j]
            j+=1
        #这里是为了将适应度划分成区间
        fitness1[i]=total
        fitness1[len(fitness1)-1]=1
```

+++ {"id": "AF2CBF8E56C645D4B9FD06AF57BFBCA4", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

(3).再产生一个0到1之间的随机数，依据随机数出现在上述哪个概率区域内来确定各个个体被选中的次数。

```{code-cell} ipython3
---
id: 438ECF7F7A9F4771A5EBFE846BC2C5A0
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
#3.选择种群中个体适应度最大的个体
def selection(population,fitness1):
    new_fitness=[]
    #单个公式暂存器
    total_fitness=sum(fitness1)
    #将所有的适应度求和
    for i in range(len(fitness1)):
        new_fitness.append(fitness1[i]/total_fitness)
    #将所有个体的适应度概率化,类似于softmax
    cumsum(new_fitness)
    #将所有个体的适应度划分成区间
    ms=[]
    #存活的种群
    population_length=pop_len=len(population)
    #求出种群长度
    #根据随机数确定哪几个能存活
 
    for i in range(pop_len):
        ms.append(random.random())
    # 产生种群个数的随机值
    ms.sort()
    # 存活的种群排序
    fitin=0
    newin=0
    new_population=new_pop=population
 
    #轮盘赌方式
    while newin<pop_len:
        if(ms[newin]<new_fitness[fitin]):
            new_pop[newin]=pop[fitin]
            newin+=1
        else:
            fitin+=1
    population=new_pop
```

+++ {"id": "C90062A5A5C946F8BC77A0C01122986B", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

4.交叉

```{code-cell} ipython3
---
id: 94F11468BC284B6C85A43D1910652969
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
def crossover(population,pc):
#pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
    pop_len=len(population)
 
    for i in range(pop_len-1):
        cpoint=random.randint(0,len(population[0]))
        #在种群个数内随机生成单点交叉点
        temporary1=[]
        temporary2=[]
 
        temporary1.extend(pop[i][0:cpoint])
        temporary1.extend(pop[i+1][cpoint:len(population[i])])
        #将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
        #然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
 
        temporary2.extend(pop[i+1][0:cpoint])
        temporary2.extend(pop[i][cpoint:len(pop[i])])
        # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
        # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
        pop[i]=temporary1
        pop[i+1]=temporary2
        # 第i个染色体和第i+1个染色体基因重组/交叉完成
```

+++ {"id": "89823B04BB634DC980C02302F73765AF", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

5.变异

```{code-cell} ipython3
---
id: BF014AD9ABA040FE837C03E4F7E5F19C
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
#step4：突变
def mutation(population,pm):
    # pm是概率阈值
    px=len(population)
    # 求出种群中所有种群/个体的个数
    py=len(population[0])
    # 染色体/个体中基因的个数
    for i in range(px):
        if(random.random()<pm):
        #如果小于阈值就变异
            mpoint=random.randint(0,py-1)
            # 生成0到py-1的随机数
            if(population[i][mpoint]==1):
            #将mpoint个基因进行单点随机变异，变为0或者1
                population[i][mpoint]=0
            else:
                population[i][mpoint]=1
```

+++ {"id": "F4ECCE4B784B43BFA11E74D91A8BB2BE", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

6.其他

```{code-cell} ipython3
---
id: E8129B36838C460489B38401BCDEF3DE
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
# 将每一个染色体都转化成十进制 max_value为基因最大值，为了后面画图用
def b2d(b,max_value,chromosome_length):
    total=0
    for i in range(len(b)):
        total=total+b[i]*math.pow(2,i)
    #从第一位开始，每一位对2求幂，然后求和，得到十进制数？
    total=total*max_value/(math.pow(2,chromosome_length)-1)
    return total
 
#寻找最好的适应度和个体
def best(population,fitness1):
 
    px=len(population)
    bestindividual=[]
    bestfitness=fitness1[0]
 
    for i in range(1,px):
   # 循环找出最大的适应度，适应度最大的也就是最好的个体
        if(fitness1[i]>bestfitness):
 
            bestfitness=fitness1[i]
            bestindividual=population[i]
 
    return [bestindividual,bestfitness]
```

+++ {"id": "775B16EDD0A5406A83A90A202E0815A9", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

7.主程序

```{code-cell} ipython3
---
id: 016F81A5CCA3460D91D8A6281BC0579E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
population_size=500
max_value=10
# 基因中允许出现的最大值
chromosome_length=10
pc=0.6
pm=0.01
 
results=[[]]
fitness1=[]
fitmean=[]
 
population=pop=species_origin(population_size,chromosome_length)
#生成一个初始的种群
 
for i in range(population_size):#注意这里是迭代500次
    function1=function(population,chromosome_length,max_value)
    fitness1=fitness(function1)
    best_individual,best_fitness=best(population,fitness1)
    results.append([best_fitness,b2d(best_individual,max_value,chromosome_length)])
     #将最好的个体和最好的适应度保存，并将最好的个体转成十进制
    selection(population,fitness1)#选择
    crossover(population,pc)#交配
    mutation(population,pm)#变异
 
results=results[1:]
results.sort()
X=[]
Y=[]
for i in range(500):#500轮的结果
    X.append(i)
    Y.append(results[i][0])
plt.plot(X,Y)
plt.show()
```

+++ {"id": "D915D53454EB439D80570A2772E5154F", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

************

+++ {"id": "B8A1D6FE80C94D3A827ECEECB075B7BB", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

## 完整代码

```{code-cell} ipython3
---
id: BD175FBA511241079ED3F20F6D2B68B9
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
# -*-coding:utf-8 -*-
#目标求解2*sin(x)+cos(x)最大值
%matplotlib notebook
import random
import math
import matplotlib.pyplot as plt
%matplotlib inline
class GA(object):
#初始化种群 生成chromosome_length大小的population_size个个体的种群
 
    def __init__(self,population_size,chromosome_length,max_value,pc,pm):
 
        self.population_size=population_size
        self.choromosome_length=chromosome_length
        # self.population=[[]]
        self.max_value=max_value
        self.pc=pc
        self.pm=pm
        # self.fitness_value=[]
 
 
 
    def species_origin(self):
        population=[[]]
        for i in range(self.population_size):
 
            temporary=[]
        #染色体暂存器
            for j in range(self.choromosome_length):
 
                temporary.append(random.randint(0,1))
            #随机产生一个染色体,由二进制数组成
 
            population.append(temporary)
            #将染色体添加到种群中
        return population[1:]
            # 将种群返回，种群是个二维数组，个体和染色体两维
 
    #从二进制到十进制
    #编码  input:种群,染色体长度 编码过程就是将多元函数转化成一元函数的过程
    def translation(self,population):
 
        temporary=[]
        for i in range(len(population)):
            total=0
            for j in range(self.choromosome_length):
                total+=population[i][j]*(math.pow(2,j))
            #从第一个基因开始，每位对2求幂，再求和
            # 如：0101 转成十进制为：1 * 20 + 0 * 21 + 1 * 22 + 0 * 23 = 1 + 0 + 4 + 0 = 5
            temporary.append(total)
        #一个染色体编码完成，由一个二进制数编码为一个十进制数
        return temporary
   # 返回种群中所有个体编码完成后的十进制数
 
 
 
#from protein to function,according to its functoin value
 
#a protein realize its function according its structure
# 目标函数相当于环境 对染色体进行筛选，这里是2*sin(x)+math.cos(x)
    def function(self,population):
        temporary=[]
        function1=[]
        temporary=self.translation(population)
        for i in range(len(temporary)):
            x=temporary[i]*self.max_value/(math.pow(2,self.choromosome_length)-10)
            function1.append(2*x*math.sin(x)+math.cos(x))
 
         #这里将sin(x)作为目标函数
        return function1
 
#定义适应度
    def fitness(self,function1):
 
        fitness_value=[]
 
        num=len(function1)
 
        for i in range(num):
 
            if(function1[i]>0):
                temporary=function1[i]
            else:
                temporary=0.0
        # 如果适应度小于0,则定为0
 
            fitness_value.append(temporary)
        #将适应度添加到列表中
 
        return fitness_value
 
#计算适应度和
 
    def sum(self,fitness_value):
        total=0
 
        for i in range(len(fitness_value)):
            total+=fitness_value[i]
        return total
 
#计算适应度斐伯纳且列表
    def cumsum(self,fitness1):
        for i in range(len(fitness1)-2,-1,-1):
        # range(start,stop,[step])
        # 倒计数
            total=0
            j=0
 
            while(j<=i):
                 total+=fitness1[j]
                 j+=1
 
            fitness1[i]=total
            fitness1[len(fitness1)-1]=1
 
 
#3.选择种群中个体适应度最大的个体
    def selection(self,population,fitness_value):
        new_fitness=[]
    #单个公式暂存器
        total_fitness=self.sum(fitness_value)
    #将所有的适应度求和
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i]/total_fitness)
    #将所有个体的适应度正则化
        self.cumsum(new_fitness)
    #
        ms=[]
    #存活的种群
        population_length=pop_len=len(population)
    #求出种群长度
    #根据随机数确定哪几个能存活
 
        for i in range(pop_len):
            ms.append(random.random())
    # 产生种群个数的随机值
    # ms.sort()
    # 存活的种群排序
        fitin=0
        newin=0
        new_population=new_pop=population
 
    #轮盘赌方式
        while newin<pop_len:
              if(ms[newin]<new_fitness[fitin]):
                 new_pop[newin]=population[fitin]
                 newin+=1
              else:
                 fitin+=1
        population=new_pop
 
#4.交叉操作
    def crossover(self,population):
#pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
        pop_len=len(population)
 
        for i in range(pop_len-1):
 
            if(random.random()<self.pc):
 
               cpoint=random.randint(0,len(population[0]))
           #在种群个数内随机生成单点交叉点
               temporary1=[]
               temporary2=[]
 
               temporary1.extend(population[i][0:cpoint])
               temporary1.extend(population[i+1][cpoint:len(population[i])])
           #将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
           #然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
 
               temporary2.extend(population[i+1][0:cpoint])
               temporary2.extend(population[i][cpoint:len(population[i])])
        # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
        # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
               population[i]=temporary1
               population[i+1]=temporary2
        # 第i个染色体和第i+1个染色体基因重组/交叉完成
    def mutation(self,population):
     # pm是概率阈值
         px=len(population)
    # 求出种群中所有种群/个体的个数
         py=len(population[0])
    # 染色体/个体基因的个数
         for i in range(px):
             if(random.random()<self.pm):
                mpoint=random.randint(0,py-1)
            #
                if(population[i][mpoint]==1):
               #将mpoint个基因进行单点随机变异，变为0或者1
                   population[i][mpoint]=0
                else:
                   population[i][mpoint]=1
 
#transform the binary to decimalism
# 将每一个染色体都转化成十进制 max_value,再筛去过大的值
    def b2d(self,best_individual):
        total=0
        b=len(best_individual)
        for i in range(b):
            total=total+best_individual[i]*math.pow(2,i)
 
        total=total*self.max_value/(math.pow(2,self.choromosome_length)-1)
        return total
 
#寻找最好的适应度和个体
 
    def best(self,population,fitness_value):
 
        px=len(population)
        bestindividual=[]
        bestfitness=fitness_value[0]
        # print(fitness_value)
 
        for i in range(1,px):
   # 循环找出最大的适应度，适应度最大的也就是最好的个体
            if(fitness_value[i]>bestfitness):
 
               bestfitness=fitness_value[i]
               bestindividual=population[i]
 
        return [bestindividual,bestfitness]
 
 
    def plot(self, results):
        X = []
        Y = []
 
        for i in range(500):
            X.append(i)
            Y.append(results[i][0])
 
        plt.plot(X, Y)
        plt.show()
 
    def main(self):
 
        results = [[]]
        fitness_value = []
        fitmean = []
 
        population = pop = self.species_origin()
 
        for i in range(500):
            function_value = self.function(population)
            # print('fit funtion_value:',function_value)
            fitness_value = self.fitness(function_value)
            # print('fitness_value:',fitness_value)
 
            best_individual, best_fitness = self.best(population,fitness_value)
            results.append([best_fitness, self.b2d(best_individual)])
        # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制,适应度函数
            self.selection(population,fitness_value)
            self.crossover(population)
            self.mutation(population)
        results = results[1:]
        results.sort()
        self.plot(results)
 
if __name__ == '__main__':
   population_size=400
   max_value=10
   chromosome_length=20
   pc=0.6
   pm=0.01
   ga=GA(population_size,chromosome_length,max_value,pc,pm)
   ga.main()
```

+++ {"id": "493D98080ACC47EE8DA5C39A7F14197E", "mdEditEnable": false, "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}}

作为对比,下面简单介绍一种基于贪婪思想的**爬山法**:

从搜索空间中随机产生邻近的点，从中选择对应解最优的个体，替换原来的个体，不断重复上述过程。


因为爬山法只对“邻近”的点作比较，所以目光比较“短浅”，常常只能收敛到离开初始位置比较近的局部最优解上面。对于存在很多局部最优点的问题，通过一个简单的迭代找出全局最优解的机会非常渺茫。（在爬山法中，袋鼠最有希望到达最靠近它出发点的山顶，但不能保证该山顶是珠穆朗玛峰，或者是一个非常高的山峰。因为一路上它**只顾上坡，没有下坡**。）

+++ {"id": "422282E2165E46B58B7449BFEA6BDFA2", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 模拟退火算法

+++ {"id": "C4901382FC7844FBA8989D637107BCC7", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

模拟退火算法 (Simulated Annealing，SA) 最早的思想是由 N. Metropolis 等人于1953年提出。1983年, S. Kirkpatrick 等成功地将退火思想引入到组合优化领域。它是**基于 Monte-Carlo 迭代求解策略的一种随机寻优算法**，其出发点是基于物理中固体物质的退火过程与一般组合优化问题之间的相似性。模拟退火算法从某一较高初温出发，伴随温度参数的不断下降,结合概率突跳特性在解空间中随机寻找目标函数的全局最优解，即在局部最优解能概率性地跳出并最终趋于全局最优。

与具有陷入局部最小值的缺点的基于梯度的方法(也就是刚才提到的爬山法)以及其他确定性搜索方法不同，SA 的主要优势在于其具有避免陷入局部最小值的能力。事实上，已经证明，如果足够的随机性与非常缓慢的冷却结合使用，模拟退火将收敛到其全局最优性。实质上，SA 是一种搜索算法，**可以视为马尔可夫链**，能够在合适的条件下收敛。

模拟退火在建模比赛中最主要的应用是模型求解，特别是最优化问题（组合优化问题），如旅行商问题 ( TSP）、最大截问题(Max Cut Problem)、0-1背包问题(Zero OneKnapsack Problem)、图着色问题(Graph Colouring Problem)、调度问题(Scheduling Problem)等等。

+++ {"id": "5E447358778A4D5F94FC3EFD508C89B5", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}


+++ {"id": "38BA73C6FB9440CF96D1D842D96C3B7F", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

求以下函数的最小值：
$$
f(x)=x^3-60x^2-4x+6,x\in [0,100]
$$

```{code-cell} ipython3
---
id: E59EB39923FF4A49BE2DBF1EA53036E6
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
x = [0.1*i for i in range(0,1000)]
y = [i**3 - 60 *i **2 - 4 * i + 6 for i in x]
plt.plot(x,y)
```

+++ {"id": "20579766D7274180B32A0E16F22AB07C", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 物理学中的退火

+++ {"id": "DAC44A1E8EB34FC58A5CA19D02E60A1B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**加温过程**：增强粒子运动，消除系统原先可能存在的非均匀态。

**等温过程**：对于与环境换热而温度不变的封闭系统，系统状态的自发变化总是朝自由能减少的方向进行，当自由能达到最小时，系统达到平衡。

**冷却过程**：使粒子热运动减弱并渐趋有序，系统能量逐渐下降，从而得到低能的晶体结构。

+++ {"id": "D16DEF0453654A8F8DD029A9BD39EFE6", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## Boltzman 概率分布

+++ {"id": "D1CAC5941BC4429BA519C163AEFB85A5", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

在统计学中，玻尔兹曼分布的表达形式为
$$
F(state) \propto  e^{-E/kT}
$$
上式中，$E$是能量状态，$kT$是玻尔兹曼常数和热力学温度的乘积。

系统的两种状态之间的玻尔兹曼分布比率称为玻尔兹曼因子，
$$
\dfrac{F(state2) }{F(state1) } = e^{-(E_1 - E_2 )/kT}
$$

需要修改

+++ {"id": "69B891342CB0476EA86E9B2CA8CAE8A0", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

- 同一个温度，分子停留在能量小状态的概率大于停留在能量大状态的概率
- 温度越高，不同能量状态对应的概率相差越小，温度足够高时，各状态对应概率基本相同。
- 随着温度的下降，能量最低状态对应概率越来越大，温度趋于0时，其状态趋于1

+++ {"id": "EFFFACDBF82A48CC805D1F20BD88E64E", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 劣解接受概率（Metropolis准则）

又称做状态接受函数，这里是模拟退火法这个名字的来源，模仿固体退火原理，随着温度的下降（迭代次数上升），能量逐渐稳定。即劣解的接受概率p逐渐下降，其公式为：
$$
\displaystyle
p=\left\{\begin{array}{ll}{1,} & {E\left(x_{new}\right) < E\left(x_{o l d}\right)} \\ {exp \left({-\dfrac{E\left(x_{n e w}\right)-E\left(x_{o l d}\right)}{T}}\right),} & {E\left(x_{n e w}\right) \geq E\left(x_{o l d}\right)}\end{array}\right.
$$

+++ {"id": "3E40BA56110D4A7CB75431C7D9425E79", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

从公式中可以看出这是求最小值时的分布，因为当新的解小于旧解时百分百接受。又可以看出当$E\left(x_{n e w}\right) \geq E\left(x_{o l d}\right)$时，$p$恒在[0, 1]之间。可以看到，随着时间推移，温度越低，对于劣解的接受概率越低。

```{code-cell} ipython3
---
id: 91C1604E070A4F539681FE82C33C5221
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

+++ {"id": "44B5784AE228408580CDF3B50031800E", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 初始化

关于初始温度，初温越大，获得高质量解的机率越大，但花费较多的计算时间，原理可以从上面的图中或者结合下面温度随时间变化进行理解、解释，温度越高对于劣解的接受能力越大（搜索范围越大），下降速度越慢（求解速度下降）。

关于初始解如何设定一般有以下方法：

（1）按照某一概率分布抽样一组$K$个解，以$K$个解的目标值的方差为初温（下面使用的时正态分布）。

（2）随机产生一组$K$个状态，确定两两状态间的最大目标差值，根据差值，利用一定的函数确定初温。

（3）利用经验公式。

+++ {"id": "2FC1032081E549F883188CDEBB220E87", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 温度更新

**更新函数1:对数函数**
$$
T(t+1)=\frac{T_{0}}{\ln (1+t)}
$$

如果使用这种温度更新，不管初始温度如何，都有一个先加热后散热的过程。即先提高搜索范围，再慢慢的减少搜索范围。

```{code-cell} ipython3
---
id: 7D8C50B40C0C4ABF85311947227849DD
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
T0_list = [1000,500,200,100]
for T0 in T0_list:
    T_list = []
    t_list = [i for i in range(1,40)]
    T_list.append(T0)
    for t in t_list:
        T_list.append(T0/np.log(t+1))
    plt.plot([i for i in range(40)],T_list,label = 'T0 = '+ str(T0))
plt.legend()
```

+++ {"id": "E449D9ADAF3C4E2F8B8CA65256809885", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**更新函数2：线性函数**
$$
T(t+1)= \alpha T(t), \alpha \in (0,1)
$$

指数衰减，通常$0.8 < \alpha <0.99$。

```{code-cell} ipython3
---
id: 5E3794B983434A969F5ECA0CC8ABD67A
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

+++ {"id": "B75CF1294BE84D0B9BE193AA0B21DE5D", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**更新函数3**：
$$
T(t)=  \frac{T_{0}}{1+t}
$$

t是迭代次数。按照反比例函数衰减，类似更新函数2。

```{code-cell} ipython3
---
id: F3C233757DAE41D48C875D2A7FB10C90
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

+++ {"id": "2B6D8A7B63394A708FAA926BF1068FD6", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 算法步骤

+++ {"id": "3A4C148E09084AA88C622B73DD6F8521", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/q3w7sz5x3e.png?imageView2/0/w/960/h/960)

+++ {"id": "A430F0C32E1943BB80CC4B99D2BC0029", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**步骤**

1. 初始化温度$T$，初始解状态$S$，每个温度$t$下的迭代次数$L$；

2. 当$k = 1,2,\cdots,L$时，进行3~6；

3. 对当前解进行变换得到新解$S'$（例如对某些解中的元素进行互换，置换）；

4. 计算增量$\Delta t'=C(S')-C(S)$，其中$C(S)$为评价函数；

5. 若$\Delta t'<0$则接受$S'$作为新的当前解，否则以概率$e^{-\Delta t'/(KT)}$接受$S′$作为新的当前解($K$为玻尔兹曼常数，数值为：$K=1.380650× 10^{-23} J/K$）；（实际上，为了简便运算，而又不失一般性，通常忽略$K$，令$K=1$）

6. 如果满足终止条件则输出当前解作为最优解，结束程序；

7. 减小$T$，转到第2步，直到$T$小于初始设定的阈值。

+++ {"id": "472E86D5B7A64B619E3BCCD16DCE3FF6", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### step0初始化

### step1产生新解

为便于后续的计算和接受，减少算法耗时，通常选择由当前新解经过简单地变换即可产生新解的方法，如对构成新解的全部或部分元素进行置换、互换等，注意到产生新解的变换方法决定了当前新解的邻域结构，因而对冷却进度表的选取有一定的影响。


### step2计算目标函数差
因为目标函数差仅由变换部分产生，所以目标函数差的计算最好按增量计算。事实表明，对大多数应用而言，这是计算目标函数差的最快方法。

### step3判断新解是否被接受

判断的依据是一个接受准则，最常用的接受准则是 Metropolis 准则

### step4当新解被确定接受时，用新解代替当前解

当新解被确定接受时，用新解代替当前解，这只需将当前解中对应于产生新解时的变换部分予以实现，同时修正目标函数值即可。此时，当前解实现了一次迭代。可在此基础上开始下一轮试验。而当新解被判定为舍弃时，则在原当前解的基础上继续下一轮试验。

### step5温度更新，下一轮循环






+++ {"id": "FE02C5C7AD754C8FB49176546D39EBDC", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 小结

模拟退火其实也是一种Greedy算法，但是它的搜索过程引入了随机因素。**模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解。**以上图为例，模拟退火算法在搜索到局部最优解B后，会以一定的概率接受向右继续移动。也许经过几次这样的不是局部最优的移动后会到达B 和C之间的峰点，于是就跳出了局部最小值B。



![Image Name](https://cdn.kesci.com/upload/image/q3w7vbb61r.png?imageView2/0/w/960/h/960)

+++ {"id": "1EB77FC2389E4A759AC8440903525023", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

模拟退火算法来源于固体退火原理，将固体加温至充分高，再让其徐徐冷却，加温时，固体内部粒子随温升变为无序状，内能增大，而徐徐冷却时粒子渐趋有序，在每个温度都达到平衡态，最后在常温时达到基态，内能减为最小。

根据 Metropolis 准则，粒子在温度$T$时趋于平衡的概率为$e^{-\frac{\Delta E}{kT}}$，其中$E$为温度$T$时的内能，$\Delta E$为其改变量，$k$为 Boltzmann 常数。

用固体退火模拟组合优化问题，将内能$E$模拟为目标函数值$f$，温度$T$演化成控制参数$t$，即得到解组合优化问题的模拟退火算法：

由初始解i和控制参数初值t开始，对当前解重复“产生新解→计算目标函数差→接受或舍弃”的迭代，并逐步衰减$t$值，算法终止时的当前解即为所得近似最优解，这是基于蒙特卡罗迭代求解法的一种启发式随机搜索过程。退火过程由冷却进度表(Cooling Schedule)控制，包括控制参数的初值$t$及其衰减因子$\Delta t$、每个$t$值时的迭代次数$L$和停止条件$S$。

+++ {"id": "1608AF4F185C405787B5BBDA4DD44E49", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

让我们实现例1、2的模拟退火算法

特别关注：
初始化方法、
温度更新方法、
产生新解方法

+++ {"id": "3428309D92374BD0AC99BFF81DB1561B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 例1代码实现：函数寻优（最小值）

```{code-cell} ipython3
---
id: 6D556C98F8F84CB282062DEE866215E8
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
```

```{code-cell} ipython3
---
id: 38BF7E5097B2437985669236185679EF
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---

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
```

+++ {"id": "FA820E45B3154DDB98ADDE22F0473518", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 例2代码实现：求解TSP

```{code-cell} ipython3
---
id: 2959271AB31146F1B2D908DB3E7DF719
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
solutionnew = np.arange(num)
solutionnew
```

```{code-cell} ipython3
---
id: A13FECAD58DA44468CB345F0A2041AB7
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
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
    print (t) #程序运行时间较长，打印t来监视程序进展速度
#用来显示结果
plt.plot(np.array(result))
plt.ylabel("bestvalue")
plt.xlabel("t")
plt.show()
```

+++ {"id": "B94A8973F78C40298A53CDE312932C9B", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 启发式优化方法总结

+++ {"id": "9CF189793DE442198C38E294B2D16998", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

**遗传算法**

优点：
 - 可以解决复杂的实际问题
 - 求解思路比较简单（与纯数学方法比较）
 - 算法计算时间与问题相关性不大

缺点： 	
 - 算法要根据问题调整
 - 求出的解是满意解，只能接近最优解
 - 存在算法不适用的情况，并且难以判断原因



**模拟退火算法**

迭代搜索效率高，并且可以并行化；

算法中有一定概率接受比当前解较差的解，因此一定程度上可以跳出局部最优；

算法求得的解与初始解状态S无关，因此有一定的鲁棒性；

具有渐近收敛性，已在理论上被证明是一种以概率l收敛于全局最优解的全局优化算法。
