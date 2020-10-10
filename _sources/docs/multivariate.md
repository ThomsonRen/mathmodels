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


# 多元分析

## 多元分析简介


多元分析（Multivariate Analyses）是多变量的统计分析方法，是数理统计中应用广泛的一个重要分支，其内容庞杂，视角独特，方法多样，深受工程技术人员的青睐和广泛使用，并在使用中不断完善和创新。由于变量的相关性，不能简单地把每个变量的结果进行汇总，这是多变量统计分析的基本出发点。

## 聚类分析


将认识对象进行分类是人类认识世界的一种重要方法，比如有关世界的时间进程的研究，就形成了历史学，也有关世界空间地域的研究，则形成了地理学。又如在生物学中，为了研究生物的演变，需要对生物进行分类，生物学家根据各种生物的特征，将它们归属于不同的界、门、纲、目、科、属、种之中。

通常，人们可以凭经验和专业知识来实现分类。而聚类分析（Cluster Analyses）作为一种定量方法，将从数据分析的角度，给出一个更准确、细致的分类工具。


### 相似度度量

#### 样本的相似度度量

要用数量化的方法对事物进行分类，就必须用数量化的方法描述事物之间的相似程度。一个事物常常需要用多个变量来刻画。如果对于一群有待分类的样本点需用$p$个变量描述，则每个样本点可以看成是 $R^{p}$ 空间中的一个点。因此，很自然地想到可以用距离来度量样本点间的相似程度

记$\Omega$ 是样本点集，距离 $d(⋅,⋅)$ 是 $\Omega \times \Omega \rightarrow R^{+}$的一个函数，满足条件：



- $d(x, y) \geq 0, \quad x, y \in \Omega$
- $d(x, y)=0$ 当且仅当 $x=y$
- $d(x, y)=d(y, x), x, y \in \Omega$
- $d(x, y) \leq d(x, z)+d(x, y), x, y, z \in \Omega$

这一距离的定义是我们所熟知的，它满足正定性，对称性和三角不等式。在聚类分析中，对于定量变量，最常用的是 Minkowski 距离

$$
d_{q}(x, y)=\left[\sum_{k=1}^{p}\left|x_{k}-y_{k}\right|^{q}\right]^{\frac{1}{q \mid}}, \quad q>0
$$


当 $q=1,2$ 或 $q \rightarrow+\infty$ 时， 则分别得到

1)绝对值距离
$$
d_{1}(x, y)=\sum_{k=1}^{q}\left|x_{k}-y_{k}\right|
$$
2）欧式距离
$$
d_{2}(x, y)=\left[\sum_{k=1}^{p}\left|x_{k}-y_{k}\right|^{2}\right]^{\frac{1}{2}}
$$
3）chebyshev距离
$$
d_{\infty}(x, y)=\max _{1 \leq k \leq p}\left|x_{k}-y_{k}\right|
$$

最常用的是欧氏距离，它的主要优点是当坐标轴进行正交旋转时，欧氏距离是保持不变的。

值得注意的是在采用 Minkowski 距离时，一定要采用相同量纲的变量。如果变量的量纲不同，测量值变异范围相差悬殊时，建议首先进行数据的标准化处理，然后再计算距离。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D6FC5810910A419E871249708EF887B5", "mdEditEnable": false}

#### 1.1.2 类与类间的相似性度量

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "08B89FDBDF2E40AFB41B6207C01B2590", "mdEditEnable": false}

如果有两个样本类$G1$和$G2$ ，我们可以用下面的一系列方法度量它们间的距离：

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "68E0B8CB024243E58E9D9962D7E6D218", "mdEditEnable": false}

1）最短距离法（nearest neighbor or single linkage method）
$$
D\left(G_{1}, G_{2}\right)=\min _{x_{i} \in G_{j} \in G_{2}}\left\{d\left(x_{i}, y_{j}\right)\right\}
$$
它的直观意义为两个类中最近两点间的距离。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "DB7A0CC0CB1A4BF0B68C6DD94E838736", "mdEditEnable": false}

2）最长距离法（farthest neighbor or complete linkage method）
$$
D\left(G_{1}, G_{2}\right)=\max _{x_{i} \in G_{j} \atop y_{j} \in G_{2}}\left\{d\left(x_{i}, y_{j}\right)\right\}
$$
它的直观意义为两个类中最远两点间的距离。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "75AC7A89227D476E8995D1595D9A79F2", "mdEditEnable": false}

3）重心法（centroid method）
$$
D\left(G_{1}, G_{2}\right)=d(\bar{x}, \bar{y})
$$
$
\text { 其中 } \bar{x}, \bar{y} \text { 分别为 } G_{1}, G_{2} \text { 的重心。 }
$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B2DCD14ED59145039900EF494330D88D", "mdEditEnable": false}

4）类平均法（group average method）
$$
D\left(G_{1}, G_{2}\right)=\frac{1}{n_{1} n_{2}} \sum_{x_{i} \in G_{1} x_{j} \in G_{2}} d\left(x_{i}, x_{j}\right)
$$
它等于 $G_{1}$ ,$G_{2}$ 中两两样本点距离的平均，式中$n_1$,$n_2$ 分别为  $G_{1}$ ,$G_{2}$  中的样本点个数。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "45F7470E0FE14045AAD1303F6D605E3F", "mdEditEnable": false}

5）离差平方和法（sum of squares method）
若记
$$
\begin{array}{l}D_{1}=\sum_{x_{i} \in G_{1}}\left(x_{i}-\bar{x}_{1}\right)^{T}\left(x_{i}-\bar{x}_{1}\right), \quad D_{2}=\sum_{x_{j} \in G_{2}}\left(x_{j}-\bar{x}_{2}\right)^{T}\left(x_{j}-\bar{x}_{2}\right) \\ D_{12}=\sum_{x_{k} \in G_{i} \cup G_{2}}\left(x_{k}-\bar{x}\right)^{T}\left(x_{k}-\bar{x}\right)\end{array}
$$
其中
$$
\bar{x}_{1}=\frac{1}{n_{1}} \sum_{x_{i} \in G_{1}} x_{i}, \quad \bar{x}_{2}=\frac{1}{n_{2}} \sum_{x_{j} \in G_{2}} x_{j}, \quad \bar{x}=\frac{1}{n_{1}+n_{2}} \sum_{x_{k} \in G_{1} \cup G_{2}} x_{k}
$$
则定义
$$
D\left(G_{1}, G_{2}\right)=D_{12}-D_{1}-D_{2}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B0C9CAFCC2CF43B0854BF0F741119AD3", "mdEditEnable": false}

事实上，若$G_{1}, G_{2}$ 内部点与点距离很小，则它们能很好地各自聚为一类，并且这两类又能够充分分离（即$D_{12}$很大），这时必然有 $D=D_{12}-D_{1}-D_{2}$ 很大。因此，按定义可以认为，两类$G_{1}, G_{2}$之间的距离很大。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "AE87104023BA459EB1192492B6464F65", "mdEditEnable": false}

### 系统聚类法

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "7146410F20D84F888373A98D5137DFBB", "mdEditEnable": false}

### 1.2.1系统聚类法特点

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "119C1A8CD7F9407B868334C44CADCAF7", "mdEditEnable": false}

系统聚类法是聚类分析方法中最常用的一种方法。它的优点在于可以指出由粗到细的多种分类情况，典型的系统聚类结果可由一个聚类图展示出来。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B61F7ED8FE624434977D7CDB5B202D15", "mdEditEnable": false}

### 1.2.2 系统聚类法举例

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "1C41EDDF27DD43838B3584BC5B5F87DF", "mdEditEnable": false}

例如，在平面上有 7 个点 $w_1,w_2,w_3,w_4,w_5,w_6,w_7$（如图 1（a）），可以用聚类图（如图 1（b））
来表示聚类结果。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2460D91672B6492FB38DC9AC1F5C3D46", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq80t68hv.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "DD689303248C4C398F9183CCF473C4C5", "mdEditEnable": false}

记 $\Omega=\left\{w_{1}, w_{2}, \cdots, w_{7}\right\}$，


聚类结果如下：当距离为 $f_5$时，分为一类

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "218ED61B1F914E1EA072E04870E87A3F", "mdEditEnable": false}

$$
G_{1}=\left\{w_{1}, w_{2}, w_{3}, w_{4}, w_{5}, w_{6}, w_{7}\right\}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A4250E10FFBE45249D13BA0A49CD4B78", "mdEditEnable": false}

距离值为 $f_4$分为两类：
$$
G_{1}=\left\{w_{1}, w_{2}, w_{3}\right\}, \quad G_{2}=\left\{w_{4}, w_{5}, w_{6}, w_{7}\right\}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "4FDF80F7DEA749FD982CA4BACE4F1C1C", "mdEditEnable": false}

距离值为$f_3$分为三类：
$$
G_{1}=\left\{w_{1}, w_{2}, w_{3}\right\}, \quad G_{2}=\left\{w_{4}, w_{5}, w_{6}\right\}, \quad G_{3}=\left\{w_{7}\right\}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E9BC1D96F85F423385115859F515CBE5", "mdEditEnable": false}

距离值为 $f_2$分为四类：
$$
G_{1}=\left\{w_{1}, w_{2}, w_{3}\right\}, \quad G_{2}=\left\{w_{4}, w_{5}\right\}, \quad G_{3}=\left\{w_{6}\right\}, \quad G_{4}=\left\{w_{7}\right\}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "8ADB0FE1D29A47969D37E0A54DED42BB", "mdEditEnable": false}

距离值为 $f_1$分为六类：
$$
G_{1}=\left\{w_{4}, w_{5}\right\}, G_{2}=\left\{w_{1}\right\}, G_{3}=\left\{w_{2}\right\}, G_{4}=\left\{w_{3}\right\}, G_{5}=\left\{w_{6}\right\}, G_{6}=\left\{w_{7}\right\}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "49C125A3846F4300858C13F6AB0B75DA", "mdEditEnable": false}

距离小于 $f_1$分为七类，每一个点自成一类.

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "5CDBBF99856B418B8C70911C5A2AEBCF", "mdEditEnable": false}

<b>怎样才能生成这样的聚类图呢？步骤如下：<b>

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "1C3C50D08B5A4594A1949DA868A21F0C", "mdEditEnable": false}

设$\Omega=\left\{w_{1}, w_{2}, \cdots, w_{7}\right\}$

(1）计算$n$ 个样本点两两之间的距离$\left\{d_{i j}\right\}$ ，记为矩阵$D=\left(d_{i j}\right)_{n \times n}$ ；

(2）首先构造$n$ 个类，每一个类中只包含一个样本点，每一类的平台高度均为零；

(3）合并距离最近的两类为新类，并且以这两类间的距离值作为聚类图中的平台高
度；

(4）计算新类与当前各类的距离，若类的个数已经等于 1，转入步骤 (5），否则，回到步骤 (3）；

(5）画聚类图；

(6）决定类的个数和类。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "37AD1A82409F4B0895343AD6635B9F66", "mdEditEnable": false}

### 1.2.3 最短距离法与最长距离法

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "06E50B10FFA54419ABEE2A2F5E0E20D5", "mdEditEnable": false}

如果使用最短距离法来测量类与类之间的距离，即称其为系统聚类法中的最短距离法（又称最近邻法），最长距离法同理。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "71AE151952F64A269943D7258BC14DE0", "mdEditEnable": false}

## 1.3变量聚类法

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E22BE169CA40447288F984FD872B9E78", "mdEditEnable": false}

在实际工作中，变量聚类法的应用也是十分重要的。在系统分析或评估过程中，为避免遗漏某些重要因素，往往在一开始选取指标时，尽可能多地考虑所有的相关因素。而这样做的结果，则是变量过多，变量间的相关度高，给系统分析与建模带来很大的不
便。因此，人们常常希望能研究变量间的相似关系，按照变量的相似关系把它们聚合成若干类，进而找出影响系统的主要因素。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "F242A056FCB54F268A5B808BDB646098", "mdEditEnable": false}

### 1.3.1 变量相似性度量

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "37074DDC650047FE915B8EA0E17CA1B7", "mdEditEnable": false}

在对变量进行聚类分析时，首先要确定变量的相似性度量，常用的变量相似性度量有两种。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "DB7A4646B125471690009469F2662AC9", "mdEditEnable": false}

1）相关系数 

 记变量$x_{j}$ 的取值 $\left(x_{1 j}, x_{2 j}, \cdots, x_{n j}\right)^{T} \in R^{n}(j=1,2, \cdots, m)$则可以用两变量$x_j$与$x_k$的样本相关系数作为它们的相似性度量

$$
r_{j k}=\frac{\sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)\left(x_{i k}-\bar{x}_{k}\right)}{\left[\sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)^{2} \sum_{i=1}^{n}\left(x_{i k}-\bar{x}_{k}\right)^{2}\right]^{\frac{1}{2}}}
$$


<b>在对变量进行聚类分析时，利用相关系数矩阵是最多的。</b>


2）夹角余弦 
 也可以直接利用两变量$x_j$与$x_k$的夹角余弦$r_{jk}$ 来定义它们的相似性度量，有
$$
r_{j k}=\frac{\sum_{i=1}^{n} x_{i j} x_{i k}}{\left(\sum_{i=1}^{n} x_{i j}^{2} \sum_{i=1}^{n} x_{i k}^{2}\right)^{\frac{1}{2}}}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2510641F4EE640AD9F2174C54A26FDF1", "mdEditEnable": false}

各种定义的相似度量均应具有以下两个性质：
$$
\begin{array}{l}\text { a) }\left|r_{j k}\right| \leq 1, \text { 对于一切 } j, k \text { ; } \\ \text { b) } r_{j k}=r_{k j}, \text { 对于一切 } j, k \text { 。 }\end{array}
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "DECE01E3D52F401EB95C15F8A082AF64", "mdEditEnable": false}

<b>$\left|r_{j k}\right|$越接近1，$x_j$与$x_k$越相关或越相似。$\left|r_{j k}\right|$越接近零， $x_j$与$x_k$的相似性越弱。</b>

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "15C231546D0E44D4A59B79862296740E", "mdEditEnable": false}

### 1.3.2 变量聚类法

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D1E361AE9604460F8C8CE7BCCE8DC7CC", "mdEditEnable": false}

类似于样本集合聚类分析中最常用的最短距离法、最长距离法等，变量聚类法采用了与系统聚类法相同的思路和过程。在变量聚类问题中，常用的有最大系数法、最小系数法等。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "AA632F66D6534BBF9CEA6F35EA891339", "mdEditEnable": false}

1）最大系数法 

 在最大系数法中，定义两类变量的距离为
$$
R\left(G_{1}, G_{2}\right)=\max _{x_{j} \in G_{1} \atop x_{k} \in G_{2}}\left\{r_{j k}\right\}
$$
这时，$R\left(G_{1}, G_{2}\right)$等于两类中最相似的两变量间的相似性度量值。

2）最小系数法 

 在最小系数法中，定义两类变量的距离为
$$
R\left(G_{1}, G_{2}\right)=\min _{x_{j} \in G_{1} \atop x_{k} \in G_{2}}\left\{r_{j k}\right\}
$$
这时，$R\left(G_{1}, G_{2}\right)$等于两类中相似性最小的两变量间的相似性度量值。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "852844F436F84B028895B0F7556DBFB2", "mdEditEnable": false}

### 1.3.3变量聚类法应用举例

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "22389D3B984D4AE9A982A3AD1EDE29B5", "mdEditEnable": false}

在服装标准制定中，对某地成年女子的各部位尺寸进行了统计，通过14个部位的测量资料，获得各因素之间的相关系数表

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "457266C5BFD4485AA0986E72B986171B", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq81boanz.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "1F57CF28545D4058B0C4AB0DC8B05937", "mdEditEnable": false}

其中 $x_1$ − 上体长，$x_2$ − 手臂长，$x_3$ − 胸围，$x_4$ − 颈围，$x_5$ − 总肩围，$x_6$ − 总胸宽，$x_7$ −
后背宽，$x_8$ − 前腰节高，$x_9$ −后腰节高，$x_{10}$ −总体长，$x_{11}$ − 身高，$x_{12}$ −下体长，$x_{13}$ −
腰围， $x_{14}$ −臀围。用最大系数法对这14个变量进行系统聚类，分类结果如图下。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B80662A3E8AE4724B36168ED906AE0AB", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq81mbtb8.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "6C0FDBCE567C46D988E265B174D83D41", "mdEditEnable": false}

可以看出，人体的变量大体可以分为两类：一类反映人高、矮的变量，如上体长，手臂长，前腰节高，后腰节高，总体长，身高，下体长；另一类是反映人体胖瘦的变量，如胸围，颈围，总肩围，总胸宽，后背宽，腰围，臀围。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D32F78C5FC2B4FC695135C72D4DAAC6D", "mdEditEnable": false}

# 1.4系统聚类法python实例

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A60BF78E1E8646728822EF2CD7668FA2", "mdEditEnable": false}

首先还是导入各种库。

```{code-cell} ipython3
---
id: CC10805455D34726899AEB9D639EABFF
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np  
from matplotlib import pyplot as plt  
from scipy.cluster.hierarchy import dendrogram, linkage 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "DAB3875285AC441886A4E72CA879CFCD", "mdEditEnable": false}

接下来是生成数据集。我们这次用的数据集是随机生成的，数量也不多，一共15个数据点，分为两个数据簇，一个有7个数据点，另一个有8个。

之所以把数据点数量设置这么少，是因为便于看清数据分布，以及后面画图时容易看清图片的分类。代码如下。

```{code-cell} ipython3
---
id: 35AF6AC56ACF407F9A8825C1BD30DC79
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
state = np.random.RandomState(99) #设置随机状态  
a = state.multivariate_normal([10, 10], [[1, 3], [3, 11]], size=7)  #生成多元正态变量  
b = state.multivariate_normal([-10, -10], [[1, 3], [3, 11]], size=8)  
data = np.concatenate((a, b)) #把数据进行拼接 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "ACA7A86727804B01920962DAE4A7766A", "mdEditEnable": false}

这里我们设置一个随机状态，便于重复试验。然后利用这个随机状态生成两个变量a和b，这两个变量就是前面说过的数据簇，a有7个数据点，b有8个，a和b都是多元正态变量，其中a的均值向量是[10, 10]，b的均值向量是[-10, -10]，两者协方差矩阵是[[1, 3], [3, 11]]。这里要注意的是协方差矩阵要是正定矩阵或半正定矩阵。然后对a与b进行拼接，得到变量data。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A8B7E4C3FAC64F798B055A13120806D6", "mdEditEnable": false}

接下来要绘制数据点的分布。代码如下。

```{code-cell} ipython3
---
id: 84BF6BCAF071481293A7C66EB2D940CA
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np  
from matplotlib import pyplot as plt  
from scipy.cluster.hierarchy import dendrogram, linkage

state = np.random.RandomState(99) #设置随机状态  
a = state.multivariate_normal([10, 10], [[1, 3], [3, 11]], size=7)  #生成多元正态变量  
b = state.multivariate_normal([-10, -10], [[1, 3], [3, 11]], size=8)  
data = np.concatenate((a, b)) #把数据进行拼接 
#此处以下为绘制数据点分布代码
ig, ax = plt.subplots(figsize=(8,8)) #设置图片大小  
ax.set_aspect('equal') #把两坐标轴的比例设为相等  
plt.scatter(data[:,0], data[:,1])  
plt.ylim([-30,30]) #设置Y轴数值范围  
plt.xlim([-30,30])  
plt.show() 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A6EEFF9279894354808E3723540A4BFD", "mdEditEnable": false}

这里代码比较简单，不再赘述，主要说一下ax.set_aspect('equal')这行代码，因为matplotlib默认情况下x轴和y轴的比例是不同的，也就是相同单位长度的线段，在显示时长度是不一样的，所以要把二者的比例设为一样，这样图片看起来更协调更准确。

所绘制图片如上所示，从图中可以明显看到两个数据簇，上面那个数据簇大概集中在坐标点[10, 10]附近，而下面那个大概集中在[-10, -10]附近，这和我们设置的是一样的。从图中可以很明显看出，这个数据集大概可以分为两类，即上面的数据簇分为一类，下面的数据簇分为另一类。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "07A7ACF2522B45638479790889A9450A", "mdEditEnable": false}

然后是数据处理，代码如下。

```{code-cell} ipython3
---
id: F84E65F5B26548A28CA0258843597F1F
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
z = linkage(data, "average") #用average算法，即类平均法 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "660E5AFCA6A34EAB8500BFD4E3E1D4C9", "mdEditEnable": false}

数据处理只有这一行代码，非常简单，但难点也就在这。首先我们来看一下z的结果，如图下所示。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "530A5D1044CA42DB8E98EE0EEAAB76C8", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq84kinki.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "FB4BCBC1C28E4E5083ADA98B90314F47", "mdEditEnable": false}

这个结果不是很好理解。scipy官方对此有一些设定，比如该结果中第一行有4个数字，即11、13、0.14740505、2，前两个数字就是指“类”，刚开始每个点就是一个类，所以11和13这两个点就是两个类，第三个数字0.14740505就是这两个点的距离，这两个点被合并成一个类，所以这个新的类包含两个点（11和13），这也就是第四个点的数值2。

而这个新的类就被算为类15。注意这里是类15，不是第15个类，因为我们原来的数据集中有15个点，按照顺序就是类0、类1、类2...类14，因为python是从0开始，所以这里类15就是指第16个类。

z的第二行数据里，前两个数字是2和5，就是原来类2和类5，距离是0.3131184，包含2个点，这行数据和第一行类似。

然后再看第三行数据，前两个数字是10和15，就是类10与类15，类15就是前面第一行合并成的新类，其包含11和13这两个点，类15与类10的距离是0.39165998，这个数字是类11和13与类10的平均距离，因为我们这里用的算法是average，类10、11和13合并为了一个新类，其包含3个点，所以第四个数字就是3。

z中其他行的数据按照此规律以此类推。最后一行数据中，类26和27合并成一个新类，这个类包含了全部15个点，也就是这15个点最终划为了一个类，算法终止。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "EDA8B05E6AB94F4C8BE60957ACC6A737", "mdEditEnable": false}

接下来就是画图，代码如下

```{code-cell} ipython3
---
id: 3D4ECD34C4F3447B905E432E6802E22E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots(figsize=(8,8))  
dendrogram(z, leaf_font_size=14) #画图 
plt.title("Hierachial Clustering Dendrogram")  
plt.xlabel("Cluster label")  
plt.ylabel("Distance")  
plt.axhline(y=10) #画一条分类线  
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "66049BA536D94584964F0CD0874BD80D", "mdEditEnable": false}

完整代码如下

```{code-cell} ipython3
---
id: 8A01044BEBED432C82D21B12BAB9ED2E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np  
from matplotlib import pyplot as plt  
from scipy.cluster.hierarchy import dendrogram, linkage

a = state.multivariate_normal([10, 10], [[1, 3], [3, 11]], size=7)  #生成多元正态变量  
b = state.multivariate_normal([-10, -10], [[1, 3], [3, 11]], size=8)  
data = np.concatenate((a, b)) #把数据进行拼接 

fig, ax = plt.subplots(figsize=(8,8)) #设置图片大小  
ax.set_aspect('equal') #把两坐标轴的比例设为相等  
plt.scatter(data[:,0], data[:,1])  
plt.ylim([-30,30]) #设置Y轴数值范围  
plt.xlim([-30,30])  
plt.show()

z = linkage(data, "average") #用average算法，即类平均法

fig, ax = plt.subplots(figsize=(8,8))  
dendrogram(z, leaf_font_size=14) #画图 
plt.title("Hierachial Clustering Dendrogram")  
plt.xlabel("Cluster label")  
plt.ylabel("Distance")  
plt.axhline(y=10) #画一条分类线  
plt.show() 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A66B90C83B474EEE8B0237A4E177A27F", "mdEditEnable": false}

## 1.5 K-means聚类

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "3D3E882DECEF491583E7B30F142E2D1A", "mdEditEnable": false}

### 1.5.1 概述

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "CB3DE07609D94C8DBAF0DBCA201B4B0E", "mdEditEnable": false}

K-means算法是集简单和经典于一身的基于距离的聚类算法,采用距离作为相似性的评价指标，即认为两个对象的距离越近，其相似度就越大。

该算法认为类簇是由距离靠近的对象组成的，因此把得到紧凑且独立的簇作为最终目标。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "C9EC5DB973884AFA8952EA34B6BDE3F2", "mdEditEnable": false}

### 1.5.2 数学原理

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "6CD03B63BDC245E08ADFA33F3CE5A8F9", "mdEditEnable": false}

K-means通过迭代寻找k个类簇的一种划分方案，使得用这k个类簇的均值来代表相应各类样本时所得的总体误差最小。

k个聚类具有以下特点：各聚类本身尽可能的紧凑，而各聚类之间尽可能的分开。

k-means算法的基础是最小误差平方和准则,如果用数据表达式表示，假设簇划分为$\left(C_{1}, C_{2}, \ldots C_{k}\right)$，则我们的目标是最小化平方误差$E$:
$$
E=\sum_{i=1}^{k} \sum_{x \in C_{i}}\left\|x-\mu_{i}\right\|_{2}^{2}
$$
其中$\mu_{i}$是簇$C_i$的均值向量，有时也称质心，表达式为
$$
\mu_{i}=\frac{1}{\left|C_{i}\right|} \sum_{x \in C_{i}} x
$$
如果我们想直接寻求上式的最小值并不容易，这是一个NP难问题，只能采用启发式的迭代方法

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "044AEE82AF8446DE859FAD86F6209A72", "mdEditEnable": false}

### 1.5.3算法步骤

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "585D835F4E6742C684F754629B37E72C", "mdEditEnable": false}

1、 随机选取k个聚类质心点（cluster centroids）为$\mu_{1}, \mu_{2}, \ldots, \mu_{k} \in \mathbb{R}^{n}$。

2、 重复下面过程直到收敛 {

 对于每一个样例$i$，计算其应该属于的类
$$
c^{(i)}:=\arg \min _{j}\left\|x^{(i)}-\mu_{j}\right\|^{2}
$$

对于每一个类$j$，重新计算该类的质心
$$
\mu_{i}=\frac{1}{\left|C_{j}\right|} \sum_{x \in C_{j}} x
$$
}

收敛指如果新计算出来的质心和原来的质心之间的距离小于某一个设置的阈值，表示重新计算的质心的位置变化不大，趋于稳定

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "649BA954047F4F598A0E67E5561D4D20", "mdEditEnable": false}

### 1.5.4 K-means聚类过程

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "28AB055778F946D08E64FC1CF9B09D14", "mdEditEnable": false}



![Image Name](https://cdn.kesci.com/upload/image/qfq89rkjrw.jpg)


+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "CA22A706CC9741DF862C6E828CC01AA6", "mdEditEnable": false}

图a表达了初始的数据集，假设k=2。

图b中，我们随机选择了两个k类所对应的类别质心，即图中的红色质心和蓝色质心

图c表示通过求样本中所有点到这两个质心的距离，并标记每个样本的类别为和该样本距离最小的质心的类别，得到了所有样本点的第一轮迭代后所属的类别。

图d中，我们对我们当前标记为红色和蓝色的点分别求其新的质心新的红色质心和蓝色质心的位置已经发生了变动。

图e和图f重复了我们在图c和图d的过程，即将所有点的类别标记为距离最近的质心的类别并求新的质心。最终我们得到的两个类别如图f

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "58A98D46B1A448EF936149172437B712", "mdEditEnable": false}

### 1.5.5 K-means聚类python实战

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E6841205366948D88CA184F14EB62140", "mdEditEnable": false}

本案例采用二维数据集，共80个样本，有4个类。数据存在testSet.txt中

```{code-cell} ipython3
---
id: 35D55570ED364DD7925F3D22E94C065C
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import copy
import numpy as np
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))    # 将每个元素转成float类型
        dataMat.append(fltLine)
    return np.array(dataMat)

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    centroid = np.array(np.zeros((k,2)))   # 每个质心有n个坐标值，总共要k个质心，此处n
    for j in range(2):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroid[:,j] = minJ + rangeJ * np.random.rand(k)

    return centroid

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = dataSet.shape[0];counts=[]
    clusterAssments=[];centroids=[]
    clusterAssment = np.array(np.zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroid = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False
        count=0
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = np.inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroid[j,:], dataSet[i,:-1])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True;count+=1  # 如果分配发生变化，则需要继续迭代
                #print(clusterAssment[i,0],'-->',minIndex)
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典

        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[clusterAssment[:,0] == cent][:,:-1]   # 取第一列等于cent的所有列
            centroid[cent,:] = np.mean(ptsInClust, axis = 0)  # 算出这些数据的中心点

        
        #此处为坑
#         centroids.append(centroid)
#         clusterAssments.append(clusterAssment)
        if clusterChanged==True:
            centroids.append(copy.copy(centroid))
            clusterAssments.append(copy.copy(clusterAssment))
            counts.append(count)
    return centroids, clusterAssments,counts
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
datMat=loadDataSet('testSet.txt')
myCentroids,clustAssings,counts = kMeans(datMat,4)

#print("clusAssings",clustAssings)
print("counts",counts) # [58, 23, 3, 1]
import matplotlib.pyplot as plt
fig1=plt.figure(1,figsize=(15,30))
len_counts=int(len(counts)/2)+1
print(len_counts)

for i in range(len(counts)):
    
    ax=fig1.add_subplot(len_counts,2,i+1)
    s=clustAssings[i][:,0]+30
    c=clustAssings[i][:,0]+20
   
    ax.scatter(datMat[:,0],datMat[:,1],s,c) #s大小，c颜色
    ax.scatter(myCentroids[i][:,0],myCentroids[i][:,1],s=150,c='r',marker='+')

plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B8A7AAAED12644609F383508473562F4", "mdEditEnable": false}

### 1.5.6 使用sklearn包

```{code-cell} ipython3
---
id: 6A0B6400432E495689500EFDCF4CE3BA
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn import datasets 
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))    # 将每个元素转成float类型
        dataMat.append(fltLine[:-1])
    return np.array(dataMat)
X=loadDataSet('testSet.txt')
estimator = KMeans(n_clusters=4) # 构造聚类器
estimator.fit(X) # 聚类

label_pred = estimator.labels_ # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0') 
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1') 
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2') 
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='p', label='label3') 
plt.xlabel('sepal length') 
plt.ylabel('sepal width') 
plt.legend(loc=2) 
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "CF31FC298DB44D178D1C3B7A15F53413", "mdEditEnable": false}

# 2.主成分分析

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B19AF14C52DE4333A5DCCAF68B97A097", "mdEditEnable": false}

我们在作数据分析处理时，数据往往包含多个变量，而较多的变量会带来分析问题的复杂性。主成分分析（Principal components analysis，以下简称PCA）是一种通过降维技术把多个变量化为少数几个主成分的统计方法，是最重要的降维方法之一。它可以对高维数据进行降维减少预测变量的个数，同时经过降维除去噪声，其最直接的应用就是压缩数据，具体的应用有：信号处理中降噪，数据降维后可视化等。 

PCA把原先的n个特征用数目更少的m个特征取代，新的m个特征一要保证最大化样本方差，二保证相互独立的。新特征是旧特征的线性组合，提供一个新的框架来解释结果。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D5A223BE932E489F804637B4A494F1AE", "mdEditEnable": false}

**可以结合变量聚类法理解**

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "F74D1607676F43C0820A29A735AF0787", "mdEditEnable": false}

## 2.1主成分分析法基本思想:方差最大理论

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "06EF9208EDD6422C8A50A58A071C3FDF", "mdEditEnable": false}

如果用 $x_{1}, x_{2}, \cdots, x_{p}$表示 $p$ 门课程，$c_{1}, c_{2}, \cdots, c_{p}$ 表示各门课程的权重，那么加权之和就是

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "CB371209217044F181FB9D530DF9B3BF", "mdEditEnable": false}

$$s=c_{1} x_{1}+c_{2} x_{2}+\cdots+c_{p} x_{p}$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E65684A480AB49C898AC910AA6743277", "mdEditEnable": false}

我们希望选择适当的权重能更好地区分学生的成绩。每个学生都对应一个这样的综合成
绩，记为 $s_{1}, s_{2}, \cdots, S_{h}$ ，$n$ 为学生人数。如果这些值很分散，表明区分得好，即是说，
需要寻找这样的加权，能使$s_{1}, s_{2}, \cdots, s_{h}$ 尽可能的分散，下面来看它的统计定义

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "AB046F3FED244FF48EBE529870FD1BC7", "mdEditEnable": false}

设 $X_{1}, X_{2}, \cdots, X_{p}$ 表示以 $x_{1}, x_{2}, \cdots, x_{p}$为样本观测值的随机变量，如果能找到$c_{1}, c_{2}, \cdots, c_{p}$使得

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "39CCA88A31A14579AA3D790B3DF9177A", "mdEditEnable": false}

$$\operatorname{Var}\left(c_{1} X_{1}+c_{2} X_{2}+\cdots+c_{p} X_{p}\right)$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2A991E708E0A496283F08B4362777832", "mdEditEnable": false}

的值达到最大，则由于**方差**反映了数据差异的程度，因此也就表明我们抓住了这 $p$ 个变量的最大变异。当然，上式必须加上某种限制，否则权值可选择无穷大而没有意义，通常规定

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "CB87B86B577E4ADE854DB3331BF0FCE3", "mdEditEnable": false}

$$
c_{1}^{2}+c_{2}^{2}+\cdots+c_{p}^{2}=1
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "16D31D7CBA9043EF8F302E7FD2320CE5", "mdEditEnable": false}

在此约束下，求上式的最优解。由于这个解是 p − 维空间的一个单位向量，它代表一个“方向”，它就是常说的主成分方向。 
 一个主成分不足以代表原来的 p 个变量，因此需要寻找第二个乃至第三、第四主成分，第二个主成分不应该再包含第一个主成分的信息，统计上的描述就是让这两个主成分的协方差为零，几何上就是这两个主成分的方向正交。具体确定各个主成分的方法如下。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2AF87BAD0760464481F80CE3B057DFB9", "mdEditEnable": false}

设 $Z_{i}$ 表示第 $i$个主成分，$i=1,2, \cdots, p$，可设

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2885C3ABDFC547FD91B13E171F4E00B9", "mdEditEnable": false}

$$
\left\{\begin{array}{l}Z_{1}=c_{11} X_{1}+c_{12} X_{2}+\cdots+c_{1 p} X_{p} \\ Z_{2}=c_{21} X_{1}+c_{22} X_{2}+\cdots+c_{2 p} X_{p} \\ \cdots \ldots \ldots \ldots \ldots \ldots \ldots \ldots \ldots \ldots \\ Z_{p}=c_{p 1} X_{1}+c_{p 2} X_{2}+\cdots+c_{p p} X_{p}\end{array}\right.
$$

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "80E5CBA3DE01466681F65792BA4574BF", "mdEditEnable": false}

其中对每一个$i$ ，均有$c_{i 1}^{2}+c_{i 2}^{2}+\cdots+c_{i p}^{2}=1$，且$\left(c_{11}, c_{12}, \cdots, c_{1 p}\right)$ 使得 $\operatorname{Var}\left(Z_{1}\right)$的值达到最大；$\left(c_{21}, c_{22}, \cdots, c_{2 p}\right)$不仅垂直于$\left(c_{11}, c_{12}, \cdots, c_{1 p}\right)$  ，而且使 $\operatorname{Var}\left(Z_{2}\right)$的值达到最大；
$\left(c_{31}, c_{32}, \cdots, c_{3 p}\right)$ 同时垂直于$\left(c_{21}, c_{22}, \cdots, c_{2 p}\right)$和$\left(c_{11}, c_{12}, \cdots, c_{1 p}\right)$，并使$\operatorname{Var}\left(Z_{3}\right)$ 的值
达到最大；以此类推可得全部 p 个主成分，这项工作用手做是很繁琐的，但借助于计
算机很容易完成。剩下的是如何确定主成分的个数

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "4505BA8B0CBC4FB082AC84BFE19DE42A", "mdEditEnable": false}

**注意事项**

 1）主成分分析的结果受量纲的影响，由于各变量的单位可能不一样，如果各自改变量纲，结果会不一样，这是主成分分析的最大问题，回归分析是不存在这种情况的，所以实际中可以先把各变量的数据标准化，然后使用协方差矩阵或相关系数矩阵进行分析。 

 2）在实际研究中，由于主成分的目的是为了降维，减少变量的个数，故一般选取少量的主成分（不超过5或6个），只要它们能解释变异的70％～80％（称累积贡献率）就行了。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "949A143935A34E358B30832A755C6F26", "mdEditEnable": false}

**二维数据举例**

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "50AF8A6A26E045E092B2EB6FA1BA305D", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq8b4q5za.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "84DECE4B3A0C4AF8BDC02A0482BD7915", "mdEditEnable": false}

• a 二维数据经过投影，变为一维； 

• b 要尽可能保留原始信息。直观的感受就是投影之后尽量分散，点分布差异相对较大，没有相关性。（相反的极端情况是投影后聚成一团，变量间的差别小，蕴含的信息就少了）所以样本间有变化，才有更多信息，变化就是差异； 

• c 如何体现差异呢？，可量化的方差。这就需要找一个方向使得投影后它们在投影方向上的方差尽可能达到最大，即在此方向上所含的有关原始信息样品间的差异信息是最多的； 

• d 降维在线性空间中就是基变换，换一种说法就是，找到一组新的基向量，在此向量空间上进行投影。在图中原本的基向量是（0，1），（1,0），现在基向量换成$F_1$，为什么不选$F_2$呢(参考b)。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "4495F25F73284A808F167D28332119E1", "mdEditEnable": false}

## 2.2主成分分析法应用举例

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2E8ADB2B14D7491E86B0A4774AA55D35", "mdEditEnable": false}

在制定服装标准的过程中，对128名成年男子的身材进行了测量，每人测得的指标中含有这样六项：身高（x1）、坐高（x2） 、胸围（x3） 、手臂长（x4） 、肋围（x5）和腰围（x6） 。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A1C5054D0F4847188A18A9C18EF35015", "mdEditEnable": false}

### 第一步

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "42D55D60E1164F6CAE44101A6A0C4A93", "mdEditEnable": false}

对原始数据标准化（减去对应变量的均值，再除以其方差），并计算相关矩阵（或协方差矩阵）: 

下表为男子身材六项指标的样本相关矩阵

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "864432E193774F7590CA362B27EDC68A", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq8m8i0yi.png?imageView2/0/w/960/h/960)



+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "3571A32D3BC244638786F40941432F86", "mdEditEnable": false}

**协方差与相关系数**

协方差用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。 

期望值分别为$E[X]$与$E[Y]$的两个实随机变量X与Y之间的协方差$Cov(X,Y)$定义为：

$$
\begin{aligned} \operatorname{Cov}(X, Y) &=E[(X-E[X])(Y-E[Y])] \\ &=E[X Y]-2 E[Y] E[X]+E[X] E[Y] \\ &=E[X Y]-E[X] E[Y] \end{aligned}
$$
从直观上来看，协方差表示的是两个变量总体误差的期望。

定义$\rho_{X Y}$称为随机变量X和Y的(Pearson)相关系数
$$
\rho_{X Y}=\frac{\operatorname{Cov}(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}
$$

（此处相关系数定义和变量聚类法中的定义是一致的，只是表示形式不同）

协方差/相关矩阵，其是由矩阵各行与各列间的协方差/相关系数组成的。也就是说，协方差/相关矩阵第i行第j列的元素是原矩阵第i列和第j列的协方差/相关系数。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "534A2CF9BAD2434D9CB1A44FFA72D628", "mdEditEnable": false}

### 第二步

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "FD158109770947B680848B1B32BA7CA1", "mdEditEnable": false}

计算相关矩阵的特征值及特征向量。 

下表为前三个特征值、特征向量以及贡献率

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "FA0C5F735B114AB4B0E6A0C3503E2342", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq8c0oj0p.png?imageView2/0/w/960/h/960)

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "352E2E7E57804BADB693DBC75B249AF5", "mdEditEnable": false}

特征值从大到小排列，特征向量和特征值对应从大到小排列。前三个主成分分别为:($x_{i}^{*}$是标准化后变量)

 $$y_1= 0.469x_{1}^{*}+0.404x_{2}^{*}+0.394x_{3}^{*}+0.408x_{4}^{*}+0.337x_{5}^{*}+0.427x_{6}^{*}$$

 $$y_2=−0.365x_{1}^{*}−0.397x_{2}^{*}+0.397x_{3}^{*}−0.365x_{4}^{*}+0.569x_{5}^{*}+0.308x_{6}^{*}$$

 $$y_3=−0.092x_{1}^{*}+0.613x_{2}^{*}−0.279x_{3}^{*}−0.705x_{4}^{*}+0.164x_{5}^{*}+0.119x_{6}^{*}$$


+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "58F92C9F02EE488DB350A2DE9323B8ED", "mdEditEnable": false}

**贡献率计算公式**

总方差中属于第$i$主成分$y_i$的比例为
$$
\frac{\lambda_{i}}{\sum_{i=1}^{p} \lambda_{i}}
$$
称为主成分$y_i$的贡献率。累计贡献率就是多个主成分贡献率的加和，$\lambda_{i}$为特征值

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "4DE96046162045BF890ADDA58DDA4ADC", "mdEditEnable": false}

### 第三步

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "1D434E55C943487E8AD05AE4C804D236", "mdEditEnable": false}

根据累计贡献率(一般要求累积贡献率达到85%)可考虑取前面两个或三个主成分。

解释主成分。观察系数发现第一主成分系数多为正数，且变量都与身材大小有关系，称第一主成分为（身材）大小成分；类似分析，称第二主成分为形状成分（或胖瘦成分），称第三主成分为臂长成分。

**（结合一定的经验和猜想，解释主成分，不是所有的主成分都能被合理的解释）**

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D1E370330810490582F0ED6A3837B601", "mdEditEnable": false}

### PCA算法步骤总结

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "8F7302BFF94144CEB2B39698A245EC94", "mdEditEnable": false}

设有$m$条$n$维数据即$m$个样本

1.对原始数据标准化（减去对应变量的均值，再除以其方差）

2.求出自变量的协方差矩阵（或相关系数矩阵）；

3.求出协方差矩阵（或性关系数矩阵）的特征值及对应的特征向量；

4.将特征向量按对应特征值大小从上到下按行排列成矩阵；

5.利用前k个特征值和特征向量计算出前k个主成分；

6.将主成分用于**回归（主成分回归）、评估正态性、寻找异常值，以及通过方差接近于零的主成分发现原始变量间的多重共线性关系**等。


+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "54657CE07D9C498A949A692041158AA9", "mdEditEnable": false}

## 2.3主成分分析python实例

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "C79E474F17C4448483EF9246F4186427", "mdEditEnable": false}

这里用一个2维的数据来说明PCA，选择2维的数据是因为2维的比较容易画图。数据如下

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "BC26841C1DD2408AB39478F0E4D69276", "mdEditEnable": false}


![Image Name](https://cdn.kesci.com/upload/image/qfq8o0zvp6.jpg?imageView2/0/w/960/h/960)

```{code-cell} ipython3
---
id: 23D6DD1798D04745855D6AE563C4A4B8
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E9BB926AC841445896A8C188A2C222B3", "mdEditEnable": false}

### Step 1:求平均值以及做normalization

```{code-cell} ipython3
---
id: 88D4130AC94C4A41854F29048BAEB1BC
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
#此处以上为上一步

mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])
#画个图看分布情况
import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_aspect(1)
plt.plot(scaled_x,scaled_y,'o') 
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "ED6042C4FF9B4CDE8365A26F65DD695D", "mdEditEnable": false}

### Step 2: 求协方差矩阵(Covariance Matrix)

```{code-cell} ipython3
---
id: 400AA2ABAE4847BA844CB7A8E11FC712
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])
#此处以上为上一步

cov=np.cov(scaled_x,scaled_y) #求协方差矩阵(Covariance Matrix)
print(cov)
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "494ADF64218C4FE7A41167A75E5BEC4A", "mdEditEnable": false}

### Step 3: 求协方差矩阵的特征根和特征向量

```{code-cell} ipython3
---
id: 01594807717E433785A15B8DE7AF31BD
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])


mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_aspect(1)
plt.plot(scaled_x,scaled_y,'o',color='blue') 


cov=np.cov(scaled_x,scaled_y) #求协方差矩阵(Covariance Matrix)

#此处以上为上一步
eig_val, eig_vec = np.linalg.eig(cov) #求协方差矩阵(Covariance Matrix)的特征根和特征向量
#控制图x、y轴显示范围
plt.plot(scaled_x,scaled_y,'o',) 
xmin ,xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
#将特征向量加到我们原来的图里：
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='red')
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "62F3949F2FDF454E9F9EE99B46525254", "mdEditEnable": false}

其中红线就是特征向量。有几点值得注意：

特征向量之间是正交的，PCA其实就是利用特征向量的这个特点，重新构建新的空间体系

如果我们将数据直接乘以特征向量矩阵，那么其实我们只是以特征向量为基底，重新构建了空间

画个图感受一下

```{code-cell} ipython3
---
id: 0D2B24684DA84285ADC11ED5F5CC6243
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])


mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])


ax = plt.gca()
ax.set_aspect(1)
plt.plot(scaled_x,scaled_y,'o') 
xmin ,xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)

cov=np.cov(scaled_x,scaled_y)
eig_val, eig_vec = np.linalg.eig(cov)
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='red')
#数据直接乘以特征向量矩阵，以特征向量为基底，重新构建了空间
new_data=np.transpose(np.dot(eig_vec,np.transpose(data)))
plt.plot(new_data[:,0],new_data[:,1],'^',color='blue')
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "54378A90D8AF4C638F9D18C43EB03697", "mdEditEnable": false}

### Step 4: 选择主要成分

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "FB2EE480FCF64C6DABBEEF7BFBF4804A", "mdEditEnable": false}

得到特征值和特征向量之后，我们可以根据特征值的大小，从大到小的选择$k$个特征值对应的特征向量。

```{code-cell} ipython3
---
id: 563D77A1F7EB48B08E61D3FAA33CD2E2
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
#使用sort函数排序
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
#从eig_pairs选取前k个特征向量就行。这里，我们只有两个特征向量，选一个最大的。
feature=eig_pairs[0][1]
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "2DD55E6BF3484933B2362EA5457AFADC", "mdEditEnable": false}

### Step 5: 转化得到降维的数据

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "A99F9E02B3F04CDE889E6C9BE8C6274C", "mdEditEnable": false}

主要将原来的数据乘以经过筛选的特征向量组成的特征矩阵之后，就可以得到新的数据了。

```{code-cell} ipython3
---
id: E9C4D169723F4F2886F79C2D8F5E880E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "B6FBC54166B54BDF86C7D37F21E8C71C", "mdEditEnable": false}

### 完整代码与总结

```{code-cell} ipython3
---
id: E766089712BB4D728EE3C84733FFC381
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])


mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])


ax = plt.gca()
ax.set_aspect(1)
plt.plot(scaled_x,scaled_y,'o') 
xmin ,xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)

cov=np.cov(scaled_x,scaled_y)
eig_val, eig_vec = np.linalg.eig(cov)
print("cov:",eig_vec[:,0][0],eig_vec[:,0][1],eig_vec[:,1][0],eig_vec[:,1][1])##正交向量

plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='red')

new_data=np.transpose(np.dot(eig_vec,np.transpose(data)))
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))
print("new data\n",new_data_reduced)

plt.plot(scaled_x,scaled_y,'o',color='red')
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')
plt.plot(new_data[:,0],new_data[:,1],'^',color='blue')
plt.plot(new_data_reduced[:,0],[1.2]*10,'*',color='green')
plt.show()
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "36D60FFA0B314754857051B791D9B720", "mdEditEnable": false}

绿色的五角星是PCA处理过后得到的一维数据，为了能跟以前的图对比，将他们的高度定位1.2.

绿色五角星是红色圆点投影到蓝色线之后形成的点。这就是PCA,通过选择特征根向量，形成新的坐标系，然后数据投影到这个新的坐标系，在尽可能少的丢失信息的基础上实现降维。

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E9DEEB037D6A490A8B1B110BBCDA8759", "mdEditEnable": false}

### 使用python sklearn做PCA

```{code-cell} ipython3
---
id: 282A70C30EF947488E06D3C655BA97B6
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
from sklearn.decomposition import PCA
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])

data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

pca=PCA(n_components=1)
pca.fit(data)
pca.transform(data)
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "E0BC47007829458CBE39DB00AA2C9A0C", "mdEditEnable": false}

结果一致！

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "13D67044F1004A60866770DA002B13AC", "mdEditEnable": false}

# 课后习题

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "1AC71B84EF7141D5B91FDC4199B7425D", "mdEditEnable": false}

1.数据集如下，使用K-means算法进行聚类（聚成三类）

```{code-cell} ipython3
---
id: 33386879A20F4AA38B97F4D92D9338D8
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn import datasets 
  
iris = datasets.load_iris() 
X = iris.data[:, :2] # #表示我们取特征空间中的2个维度

print(X.shape)
  
# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see') 
plt.xlabel('sepal length') 
plt.ylabel('sepal width') 
plt.legend(loc=2) 
plt.show() 
```

+++ {"jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "id": "D051C4DF27CE44B1A89B4D5EE65CD460", "mdEditEnable": false}

2.数据如下，使用系统聚类法进行聚类

```{code-cell} ipython3
---
id: D38A0E893D3746A09CCF2EBC169FB908
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np 
data=np.array([[  9.31556164 , 10.69767437],
 [  9.21300425 ,  9.20698909],
 [ 10.16931836 , 10.50496561],
 [  8.97948118 ,  7.58733691],
 [ 11.04203644 , 10.11493029],
 [  9.881797   , 10.62895995],
 [  9.79766011 ,  9.97217018],
 [ -9.67975098 ,-11.05412528],
 [-10.58070834 ,-12.99003422],
 [ -9.7736453  , -8.27254942],
 [ -9.43198433 , -7.54610439],
 [ -9.13994785 , -7.18378278],
 [ -9.47403457 , -8.81266627],
 [ -9.23248318 , -7.29852363],
 [-10.47829261 ,-10.30055412],
 [ 10.50596899 , -8.09211343],
 [  7.81810663 ,-17.27380884],
 [  9.38509479 ,-11.5525437 ],
 [ 11.66143844 , -4.37736872],
 [ 11.62327952 , -2.8127609 ],
 [  8.64840945 ,-15.69415165],
 [  8.82528612 ,-15.96868896],
 [ 10.87536071 , -5.98830375],
 [ 10.94598118 , -7.34968697]])
```
