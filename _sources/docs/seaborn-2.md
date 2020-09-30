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

# Seaborn-2



## 在多张图片中展示数据

+++ {"slideshow": {"slide_type": "slide"}, "id": "2CAA071F82824B548B625D86AD82B3F8", "mdEditEnable": false, "jupyter": {}, "tags": []}

正如之前`relplot()`中提到的，如果我们希望绘制多张分类的图像，只需要通过设定`row`和`col`参数即可。


```{code-cell} ipython3
---
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 导入 seaborn 并且命名为sns
sns.set(style="darkgrid") # 设置绘图格式为darkgrid
tips = sns.load_dataset('tips')
tips.head()
```


```{code-cell} ipython3
---
id: 71BAEADDB2064996A55E745C8149CA38
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.catplot(x="time", y="total_bill", hue="smoker",
            col="day", aspect=.6,
            kind="swarm", data=tips);
```

+++ {"id": "D91A95D7877E4F5A8370EF14B4DA1392", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

## 绘制数据分布

+++ {"slideshow": {"slide_type": "slide"}, "id": "68FC87B954AB4A9F8659E2AF4B2C4FF2", "mdEditEnable": false, "jupyter": {}, "tags": []}

当我们遇到一个新的数据集的时候，往往我们首先要搞清楚的就是其中每一个变量的分布。本节我们将会给大家介绍seaborn中一些用于可视化数据分布的函数。

+++ {"slideshow": {"slide_type": "slide"}, "id": "55F6EB93FF6C41DA84CA481774C67BB1", "mdEditEnable": false, "jupyter": {}, "tags": []}

首先我们导入`numpy`,`pandas`,`seaborn`,`pyplot`和`stats`。

```{code-cell} ipython3
---
id: 845CEADD99CD4E8695EBE3D1A8F837D3
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
```

```{code-cell} ipython3
---
id: 8B88898DD7C74BFC8C56F3BA0D9303E1
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set(color_codes=True)
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "CA13626FA4E1422C862B027049BEECC4", "mdEditEnable": false, "jupyter": {}, "tags": []}

### 绘制单变量分布

+++ {"slideshow": {"slide_type": "slide"}, "id": "82FD23BF312E43838FB1D8002CCD60B0", "mdEditEnable": false, "jupyter": {}, "tags": []}

在seaborn中，绘制单变量分布的最简单的函数是`displot()`,该函数默认返回一张频率分布直方图以及其对应的核密度估计曲线（KDE）。

```{code-cell} ipython3
---
id: B144D6AAF1184E7181A670A185899195
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
x = np.random.normal(size =100)
sns.distplot(x);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "A06C42E02B934D5B85B45EB4A34A0EB9", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 频率分布直方图

+++ {"slideshow": {"slide_type": "slide"}, "id": "4DA7FF92B1624B6D84CE3B8EF92E1D77", "mdEditEnable": false, "jupyter": {}, "tags": []}

seaborn中的频率分布直方图`displot()`和matplotlib中的`hist()`非常相似。不过，seaborn给出了更为高层次的调用方法，我们可以通过参数`kde`和`rug`控制直方图中kde估计和数据点标记的展示与否。

```{code-cell} ipython3
---
id: AF8A9FEED9B6406F83595CED526B7DEC
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.distplot(x, kde=True, rug=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "C73C14FDE0F44F638D3058241E399F08", "mdEditEnable": false, "jupyter": {}, "tags": []}

当绘制直方图的时候，我们经常会调整的一个参数是直方的个数，控制直方个数的参数是`bins`,如果不认为指定`bins`的取值，seaborn会根据自己的算法得到一个较为合理的直方个数，但是通过人为调整直方个数，我们往往能发现新的规律。

```{code-cell} ipython3
---
id: 12B50E95A7804F1881CE5DDC8CC9E197
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.distplot(x, bins=5, kde=False, rug=True);
```

```{code-cell} ipython3
---
id: B8B354E82EE44ACA91F0F7ED5519DE68
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.distplot(x, bins=25, kde=False, rug=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "5D96FCE4FD1142A186DBC13ABBEFA0AB", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 核密度估计Kernel density estimation

+++ {"slideshow": {"slide_type": "slide"}, "id": "A8E6FCBB654B40C6AC014B234912CAF0", "mdEditEnable": false, "jupyter": {}, "tags": []}

核密度估计是一种分布的平滑（smooth）方法，所谓核密度估计，就是采用平滑的峰值函数(“核”)来拟合观察到的数据点，从而对真实的概率分布曲线进行模拟。

```{code-cell} ipython3
---
id: FD377841911140C98440E7C67248F009
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.distplot(x, hist=False, rug=True,kde = True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "F80E5F02046D44C988B9D36938360BD3", "mdEditEnable": false, "jupyter": {}, "tags": []}

那么，我们是符合得到这样一条曲线的呢？  实际上，我们将每一个数据点用一个以其为中心的高斯分布曲线代替，然后将这些高斯分布曲线叠加得到的。

```{code-cell} ipython3
---
id: 11AE65EC70EC403182B4F52A3253CD4C
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
x = np.random.normal(0, 1, size=30)   # 生成中心在0，scale为1，30维的正态分布数据 
bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.) # 确定带宽
support = np.linspace(-4, 4, 200)  
kernels = []
for x_i in x:
    kernel = stats.norm(x_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support, kernel, color="r")
sns.rugplot(x, color=".2", linewidth=3);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "CA5EAD7F1F864EAE8E91B1317A48DD66", "mdEditEnable": false, "jupyter": {}, "tags": []}

将每一个数据转化为以其为中心的正态分布曲线以后，将其叠加，然后归一化，即可得到最终的KDE曲线。

```{code-cell} ipython3
---
id: 848BB96BA03841AB98D79B50E7FACB7E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
from scipy.integrate import trapz
density = np.sum(kernels, axis=0)
density /= trapz(density, support) # 使用梯形积分计算曲线下面积，然后归一化
plt.plot(support, density);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "65E9D0F97CFA431FA90098F5B2A81B22", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们可以通过观察，发现，使用seaborn中的`kdeplot()`我们会得到同样的曲线，或者使用`distplot(kde = True)`也有同样的效果。

```{code-cell} ipython3
---
id: 46F18BA026F54591A76C64E9BF9F03E6
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.kdeplot(x, shade=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "CB043C028FF640728881FD15C5C39BAF", "mdEditEnable": false, "jupyter": {}, "tags": []}

除了核函数，另一个影响KDE的参数是带宽(h)。带宽反映了KDE曲线整体的平坦程度，也即观察到的数据点在KDE曲线形成过程中所占的比重 — 带宽越大，观察到的数据点在最终形成的曲线形状中所占比重越小，KDE整体曲线就越平坦；带宽越小，观察到的数据点在最终形成的曲线形状中所占比重越大，KDE整体曲线就越陡峭。

```{code-cell} ipython3
---
id: 08AB023170154BA8844275BBACC0B045
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "0BE8547AF2AE40808E900E66A0D30E03", "mdEditEnable": false, "jupyter": {}, "tags": []}

通过观察以上的图像我们可以发现，由于高斯分布的引入，我们往往会扩大了变量的取值范围，我们可以通过`cut`参数控制最终图像距离最小值和最大值的距离。需要注意的是，`cut`参数仅仅是改变了图像的展示方法，对kde的计算过程没有影响。

```{code-cell} ipython3
---
id: 55A299AE22BA48CB84F1E14DB260FB3C
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.kdeplot(x, shade=True, cut=4)
sns.rugplot(x);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "A9C3A91C1DE343D69AEB5723504B6D16", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 参数分布的拟合

+++ {"slideshow": {"slide_type": "slide"}, "id": "82F9CE9FFFAA4A1589157C02CB8AF139", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们也可以使用`displot()`拟合参数分布，并且将拟合结果与实际数据的分布做对比。

```{code-cell} ipython3
---
id: E52F92561252439F81AB23ECCF83F8F9
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma); # 是用gamma分布拟合，并可视化
```

+++ {"id": "13A1CCFC781745268E05F6053873829E", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 绘制两变量之间的联合分布

+++ {"id": "15FC1A1E5FD646EB83FF615B09F7AE96", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

有的时候，我们在数据分析的时候，也会关系两个变量之间的联合概率分布关系。seaborn中给我们提供了一个非常方便的`jointplot()`函数可以实现该功能。

```{code-cell} ipython3
---
id: 50D462B3D8974A8B8925748C1E627766
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
mean = [0, 1]
cov = [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
```

```{code-cell} ipython3
---
id: 01ED7987AC624B1D9DE0E881EBE7291B
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
df.head()
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "5282F961FAA147759E9FFAB77EF4B85A", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 散点图

+++ {"slideshow": {"slide_type": "slide"}, "id": "2199CA8558584D21B9939EE7809F9A3A", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们最熟悉的绘制联合分布的方法莫过于散点图了。`jointplot()`会返回一张散点图（联合分布），并在上方和右侧展示两个变量各自的单变量分布。

```{code-cell} ipython3
---
id: 6ADF1BBE9D934318887D885206C599DE
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.jointplot(x="x", y="y", data=df);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "EC9AF5193B4D41E68E8E1770F984803E", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### Hexbin plots

+++ {"slideshow": {"slide_type": "slide"}, "id": "5E590EF77B8B44B69BDA0634CC592DA9", "mdEditEnable": false, "jupyter": {}, "tags": []}

与一维柱状图对应的二维图像称之为Hexbin plots，该图像帮助我们统计位于每一个六边形区域的数据的个数，然后用颜色加以表示，这种方法尤其对于大规模的数据更为适用。

```{code-cell} ipython3
---
id: D1BC7CB7C96342398F9D497EF5878CA4
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
x, y = np.random.multivariate_normal(mean, cov, 1000).T
sns.jointplot(x=x, y=y, kind="hex", color="k");
# with sns.axes_style("white"):
#     sns.jointplot(x=x, y=y, kind="hex", color="k");
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "24F54E905DFA429A99544D12F3F9CDEC", "mdEditEnable": false, "jupyter": {}, "tags": []}

该方法尤其适用于白色风格

```{code-cell} ipython3
---
id: B5C4587A6CD34F47939167CCEF0D14A5
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="reg", color="k");
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "E192EDB5A20946B79732B2FE9BE08DF7", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 联合分布的核密度估计

+++ {"slideshow": {"slide_type": "slide"}, "id": "99DD97BEA6D740C9828CC297A1A88F14", "mdEditEnable": false, "jupyter": {}, "tags": []}

类似于一维情况，我们在二维平面一样可以进行核密度估计。通过设置`kind = 'kde'`，我们就可以得到一个核密度估计的云图，以及两个单变量的核密度估计曲线。

```{code-cell} ipython3
---
id: 234DF654D0DE4FF19DFA3F8282311129
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.jointplot(x="x", y="y", data=df, kind="kde");
```

```{code-cell} ipython3
---
id: 20A23F66C692421689CA409E417F3469
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
with sns.axes_style("white"):
    sns.jointplot(x="x", y="y", data=df, kind="kde");
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "5D422AE67CAE48AF852C816618894B2C", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们也可以直接使用`kdeplot()`绘制二维平面上的核密度估计。而且，结合面向对象的方法，我们还可以把新的绘图加入到已有的图片上。

```{code-cell} ipython3
---
id: 99A575F59F9B4BC18BDF1AF45D6B6450
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "37AFF1DC18644447850FC79E2AD9308A", "mdEditEnable": false, "jupyter": {}, "tags": []}

If you wish to show the bivariate density more continuously, you can simply increase the number of contour levels:


```{code-cell} ipython3
---
id: 09D48039D1BB4EA897AA158ACB45C46F
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=509, shade=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "A076984296C04A6F81205B69EE3AE182", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们还可以给图片添加新的图层，将数据的散点图绘制在原图上，包括给图片添加坐标轴标签等等。

```{code-cell} ipython3
---
id: B088B3DDC78646B58BB16CF4D4621CD2
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0.5)
g.set_axis_labels("$X$", "$Y$");
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "85F71BEA27AE4CB2B67A877E6CD884FF", "mdEditEnable": false, "jupyter": {}, "tags": []}

## 分组可视化

+++ {"slideshow": {"slide_type": "slide"}, "id": "0FF83B12593048CA87A3DF5FE2FA2D45", "mdEditEnable": false, "jupyter": {}, "tags": []}

借助于上述的双变量分布绘图方法，我们可以绘制多变量两两之间的联合分布，seaborn中实现这个功能的函数为`pairplot()`，该函数会返回一个方形的绘图窗口，在该窗口中绘制两两变量之间的关系。在对角线上，`pairplot()`会展示单变量分布。

```{code-cell} ipython3
---
id: 2DA3BF2A589C4159B55582C88AB0E0B8
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
# iris = sns.load_dataset('iris')
# sns.pairplot(iris);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "F67C524069A64BBCA6F2E9468DF51AF9", "mdEditEnable": false, "jupyter": {}, "tags": []}

### 可视化线性关系

+++ {"slideshow": {"slide_type": "slide"}, "id": "C61264A337374A5A90590EC64B1E5557", "mdEditEnable": false, "jupyter": {}, "tags": []}

许多数据集都包含了众多变量，有的时候我们希望能够将其中的一个或者几个联系起来。上一节我们讲到了seaborn中很多绘制联合分布的方法，比如`jointplot()`，本节我们进一步地，讨论变量之间线性关系的可视化。

+++ {"slideshow": {"slide_type": "slide"}, "id": "7BAD32892EA246778DB554EC84FE9614", "mdEditEnable": false, "jupyter": {}, "tags": []}

需要注意的是，seaborn并不是一个统计学的库，seaborn想要实现的是：通过运用简单的统计工具，尽可能简单而直观地给我们呈现出数据之间相互关系。有的时候，对数据有一个直观的认识，能帮助我们更好地建立模型。

```{code-cell} ipython3
---
id: 3B677C7DA9AF438A8D131E202641E5B6
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
---
id: 1F3F290285A84485A86148F06F4BD093
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set(color_codes=True)
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "CED26FFA81F54F98944FA5CB388D9C4A", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 绘制线性回归的函数

+++ {"slideshow": {"slide_type": "slide"}, "id": "1CB9BF5DFB68478ABB4D2F352C69C71F", "mdEditEnable": false, "jupyter": {}, "tags": []}

在seaborn中，有两个函数经常被用于实现线性回归，他们是`lmplot`和`regplot`。接下来我们会介绍这两个函数的异同。

+++ {"slideshow": {"slide_type": "slide"}, "id": "BBC2AC8D0BBC49BC858C2012350B58B1", "mdEditEnable": false, "jupyter": {}, "tags": []}

在最简单的情况下，两个函数均会返回一个散点图，并给出$y$关于$x$的线性回归方程以及一个95%的置信区间。


```{code-cell} ipython3
---
id: 0E4286FBD6DA4F00AE3BF98E7C400D58
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.regplot(x="total_bill", y="tip", data=tips);
```

```{code-cell} ipython3
---
id: 85BB24E06E26458F8645130AC395E7E3
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", data=tips);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "C86FF207FAA24E6E980AB1B796FD429A", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们发现，除了图的尺寸，这两个图的内容是完全一致的。

+++ {"slideshow": {"slide_type": "slide"}, "id": "E5728BBECF9A403ABEDC24031AC131D3", "mdEditEnable": false, "jupyter": {}, "tags": []}

那么，这两个函数有什么不同呢？

- `regplot()`能接受更多种形式的数据，例如numpy arrays, pandas Series, references to variables in a pandas DataFrame，而 `lmplot()`只能接受references to variables in a pandas DataFrame，也就是只能接受“tidy” data
- `regplot()` 仅仅指出 `lmplot()`的一部分参数

+++ {"slideshow": {"slide_type": "slide"}, "id": "E8238A08FF57406F845837123F132BF1", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们可以对一个离散变量和一个连续变量绘制线性回归线，不过，可视化结果往往是不尽如人意的。

```{code-cell} ipython3
---
id: A7187B80E5A7436A85ED9AC4EE59DE02
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="size", y="tip", data=tips);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "45A9CC270C3C44078763F151B0BFB382", "mdEditEnable": false, "jupyter": {}, "tags": []}

针对上面这个图像，一个选择是给每一个离散的变量增加一个随机的扰动`jitter`，使得数据的分布更容易观察，请注意，`jitter`参数的存在仅仅是改变了可视化的效果，不会影响线性回归方程。

```{code-cell} ipython3
---
id: AAC1066281F84954B83C2028362293F4
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="size", y="tip", data=tips, x_jitter=.3);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "304F90816D4F46C88FC7607A962BA7FF", "mdEditEnable": false, "jupyter": {}, "tags": []}

另一个选择是，我们直接将每一个离散类别中的所有数据统一处理，得到一个综合的趋势以及每个数据点对应的置信区间。

```{code-cell} ipython3
---
id: FF23634F6D5E4D9281FF1C8773C76DEC
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "EA2417A3D2504F2181981B8DF1563CC7", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 拟合其他形式的模型

+++ {"slideshow": {"slide_type": "slide"}, "id": "6F32A4ED771B451B99D05EEFFA0A5847", "mdEditEnable": false, "jupyter": {}, "tags": []}

简单的线性拟合非常容易操作，也很容易理解。但是真实的数据往往不一定是线性相关的，因此我们需要考虑更多的拟合方法。

+++ {"slideshow": {"slide_type": "slide"}, "id": "7D41A06D3FF8489084D8B53093BF7F76", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们这里使用的是 The Anscombe’s quartet dataset，在这个数据集中，不同形式的数据会得到同样的一个回归方程，但是拟合效果却是不同的。

首先我们来看第一个

```{code-cell} ipython3
---
id: DF9EEBE3600E49128785247385B35867
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
anscombe = sns.load_dataset("anscombe")
# anscombe = pd.read_csv('/home/kesci/input/Seaborn_Demo6897/anscombe.csv')
```

```{code-cell} ipython3
---
id: 4C786B740608418D8BA5ABD8D21B0D37
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "65CF227339B74153A134B956047F3E0B", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们接着来看第二个线性拟合，其拟合方程和第一个模型是一样的，但是显然其拟合效果并不好。

```{code-cell} ipython3
---
id: 245A9CD93F854D409378E87BA182563B
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "643D1FA0F9CC4D018368470C62B008B6", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们可以给`lmplot()`传入一个`order`参数，修改数据拟合的阶次，进而可以拟合非线性趋势。

```{code-cell} ipython3
---
id: 28AB6050F69349218A41CAD1962A3CFF
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "1274A4A2AC554B52ACF338C51BC58AC8", "mdEditEnable": false, "jupyter": {}, "tags": []}

接着我们来看第三个例子，在这个案例中，我们引入了一个离群点，由于离群点的存在，其拟合方程显然偏离了主要趋势。

```{code-cell} ipython3
---
id: AF6A308140154C748BA66C33FDC8D9A4
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "35EB60BAE41741778891B2173D3BAC17", "mdEditEnable": false, "jupyter": {}, "tags": []}

此时我们可以通过引入`robust`参数增强拟合的稳定性，该参数设置为True的时候，程序会自动忽略异常大的残差。

```{code-cell} ipython3
---
id: 885D0BC81FC74D8B8C6B0ECE5F1D5B9F
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "A56CDD21B20D4FA88A5E90538482B13D", "mdEditEnable": false, "jupyter": {}, "tags": []}

当y参数传入了二分数据的时候，线性回归也会给出结果，但是该结果往往是不可信的。

```{code-cell} ipython3
---
id: 2E487769387E4DF7947C40812C43BF64
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           y_jitter=.03);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "45EC8B67A054461BA68A00B65D7BA529", "mdEditEnable": false, "jupyter": {}, "tags": []}

可以考虑采取的一个方法是引入逻辑回归，从而回归的结果可以用于估计在给定的$x$数据下，$y=1$的概率

```{code-cell} ipython3
---
id: CEE8176A531E4C0C85D800B62973E3A0
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           logistic=True, y_jitter=.03);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "900EF974F6BD4EBDAF3EE06EC4077129", "mdEditEnable": false, "jupyter": {}, "tags": []}

请注意，相比如简单的线性回归，逻辑回归以及robust regression 计算量较大，同时，置信区间的计算也会涉及到bootstrap，因此如果我们想要加快计算速度的话，可以把bootstrap关掉。

+++ {"slideshow": {"slide_type": "slide"}, "id": "F446B2317F4E4B5A82931D9F5FB3E361", "mdEditEnable": false, "jupyter": {}, "tags": []}

其他拟合数据的方法包括非参数拟合中的局部加权回归散点平滑法(LOWESS)。LOWESS 主要思想是取一定比例的局部数据，在这部分子集中拟合多项式回归曲线，这样我们便可以观察到数据在局部展现出来的规律和趋势。

```{code-cell} ipython3
---
id: F69D22B2FE51425981881D6D7087F8DB
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", data=tips,
           lowess=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "A7957F2C1D8D4A8986CE3A313670949C", "mdEditEnable": false, "jupyter": {}, "tags": []}

使用`residplot()`，我们可以检测简单的线性回顾是否能够比较好地拟合原数据集。 理想情况下，简单线性回归的残差应该随机地分布在$y=0$附近。

```{code-cell} ipython3
---
id: 59B9B82715F645EDA3570F293EAD94F2
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
              scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "41D18BAF299E48658B3F8D04BD56E782", "mdEditEnable": false, "jupyter": {}, "tags": []}

如果出现了如下图所示的残差图，则说明线性回归的效果并不好。

```{code-cell} ipython3
---
id: EA8FE3189A114D3D8B4716E071157AD5
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
              scatter_kws={"s": 80});
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "E1F797C2E19F48B189F7731E69E4B9E0", "mdEditEnable": false, "jupyter": {}, "tags": []}

#### 引入第三个参数

+++ {"slideshow": {"slide_type": "slide"}, "id": "1651CF31CFC94C4B8D141EBAB795C74B", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们知道，线性回归可以帮助我们描述两个变量之间的关系。不过，一个跟有趣的问题是：“这两个变量之间的关系是否跟第三个因素有关呢？”

这时`regplot()`和`lmplot()`就有区别了。`regplot()`只能展示两个变量之间的关系，而`lmplot()`则能进一步地引入第三个因素（categorical variables）。

+++ {"slideshow": {"slide_type": "slide"}, "id": "801AB050B5584BAB94429971C2173AE9", "mdEditEnable": false, "jupyter": {}, "tags": []}

我们可以通过不同的颜色来区分不同的类别，在同一张图中绘制多个线性回归曲线：

```{code-cell} ipython3
---
id: 15809551420F4FD08485F0D24BEED30E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "E75C884041B249628359D098D413B7D0", "mdEditEnable": false, "jupyter": {}, "tags": []}

除了颜色之外，为了观察和打印方便，我们还可以引入不同的图形标记，区分不同的类别。

```{code-cell} ipython3
---
id: 1EBCFA9080614C02AD4D8E0014F0CC45
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "04F204B35DA0439591D3390990840E1A", "mdEditEnable": false, "jupyter": {}, "tags": []}

To add another variable, you can draw multiple “facets” which each level of the variable appearing in the rows or columns of the grid:

+++ {"slideshow": {"slide_type": "slide"}, "id": "E0340D8E404B4AEB9FC055E9CBAE8F13", "mdEditEnable": false, "jupyter": {}, "tags": []}

如果我们想进一步地增加维度（变成四维绘图甚至五维），我们可以增加一个`col`参数。

```{code-cell} ipython3
---
id: A8CDB2548EE240EC81F881C454114066
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
```

```{code-cell} ipython3
---
id: A45064EDC75146E883B4531FFB6370A0
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "F4E3E0D5C7114F0F8A109A8567F0BE40", "mdEditEnable": false, "jupyter": {}, "tags": []}

## 调整绘图的尺寸和形状

+++ {"slideshow": {"slide_type": "slide"}, "id": "68408D7C2BDD4F448D3DFCDC8F53F748", "mdEditEnable": false, "jupyter": {}, "tags": []}

前面我们注意到了，`regplot`和`lmplot`做出的图像基本类似，但是在图像的尺寸和形状上有所区别。

+++ {"slideshow": {"slide_type": "slide"}, "id": "26D5E1F281764D95867EA9424C256F54", "mdEditEnable": false, "jupyter": {}, "tags": []}

这是因为，`regplot`的绘图，是图层层面的绘图，这意味着我们可以同时对多个图层进行操作，然后对每个图层进行精细化的格式设置。为了控制图片尺寸，我们必须先生成一个固定尺寸的对象。

```{code-cell} ipython3
---
id: 5BC1D4CE667149BA8653076F6E3F2A37
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
f, ax = plt.subplots(figsize=(15, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "45BC9A3CC5D44D558D04D347C89E01A3", "mdEditEnable": false, "jupyter": {}, "tags": []}

与`regplot`不同的是，`lmplot`是一个集成化的命令，如果我们想要修改图片的尺寸和大小，只能通过传入参数的格式进行实现，`size`和`aspect`分别用来控制尺寸和长宽比。

```{code-cell} ipython3
---
id: 66AFCD3003FE4570B759ACA291060B10
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,height=6);
```

```{code-cell} ipython3
---
id: 355804C80A2042CE8B5910AA42206F00
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,height=6,
           aspect=1.6);
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "9A4B9500B5F7435494CC8B25CE07D030", "mdEditEnable": false, "jupyter": {}, "tags": []}



## 修改绘图风格

```{code-cell} ipython3
---
id: 10BBEBC5275E43478C17F44569BBA4FC
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

+++ {"id": "F88A59AE84514897A53E501F66AE46BD", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

让我们来定义一簇简单的正弦曲线，然后观察一下不同的绘图风格的区别。

```{code-cell} ipython3
---
id: 0C5D9A773ECA4FB197689E37D9CD7955
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
```

+++ {"id": "12D6EA8E72D04BF2B14F6412E2F17F68", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

这是matplotlib默认风格：

```{code-cell} ipython3
---
id: 32E6696E04D04C6684E8EA232455F0AB
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sinplot()
```

+++ {"id": "008CDDE1D42C47C6806DC485097F31BA", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

现在我们切换成seaborn默认风格。

```{code-cell} ipython3
---
id: 63013A5D58044C82AD8A3C51ABB238E1
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set()
sinplot()
```

+++ {"id": "8BF55980961240A98EC6D033BB414A6D", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

(Note that in versions of seaborn prior to 0.8, set() was called on import. On later versions, it must be explicitly invoked).

+++ {"id": "1548EBF97FBE4750B6693590EFD6FB84", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

Seaborn 把matplotlib中的参数分为了两类。其中第一类用来调整图片的风格（背景、线型线宽、字体、坐标轴等），第二类用来根据不同的需求微调绘图格式（图片用在论文、ppt、海报时有不同的格式需求。）

+++ {"id": "A66D1811D25046B18A6252BDB894DBED", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}


+++ {"id": "41851DD121DF4D54946573C4F402E956", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

画出令人赏心悦目的图形，是数据可视化的目标之一。我们知道，数据可视化可以帮助我们向观众更加直观的展示定量化的insight， 帮助我们阐述数据中蕴含的道理。除此之外，我们还希望可视化的图表能够帮助引起读者的兴趣，使其对我们的工作更感兴趣。


Matplotlib给了我们巨大的自由空间，我们可以根据自己的需要，任意调整图像的风格。然而，为了绘制一张上述的“令人赏心悦目”的图片，往往需要长期的绘图经验。这对新手来说时间成本无疑是非常高的。为此，seaborn也给我们集成好了一些设置好的绘图风格，使用这些内置风格，我们就能“傻瓜式”地获得美观的绘图风格。

+++ {"slideshow": {"slide_type": "slide"}, "id": "599F2010E6D84D778E330F646BF647CE", "mdEditEnable": false, "jupyter": {}, "tags": []}

### Seaborn 绘图风格

+++ {"slideshow": {"slide_type": "slide"}, "id": "6DD5F2BCC9854A9EA849DAFE758568BA", "mdEditEnable": false, "jupyter": {}, "tags": []}

在seaborn中，有五种预置好的绘图风格，分别是：`darkgrid`, `whitegrid`, `dark`, `white`和` ticks`。其中`darkgrid`是默认风格。


用户可以根据个人喜好和使用场合选择合适的风格。例如，如果图像中数据非常密集，那么使用`white`风格是比较合适的，因为这样就不会有多于的元素影响原始数据的展示。再比如，如果看图的读者有读数需求的话，显然带网格的风格是比较好的，这样他们就很容易将图像中的数据读出来。

```{code-cell} ipython3
---
id: E0222596E6F541BD84B955ED8FF7C66A
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "70C61D913AAF4F40BAF1CD252D0DBB24", "mdEditEnable": false, "jupyter": {}, "tags": []}

先来看`whitegrid`风格

```{code-cell} ipython3
---
id: 0E615A5DE81947BD995F09A8E374F0BE
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data);
```

```{code-cell} ipython3
---
id: B993072AAA034E34865B0141DD28EBE6
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("whitegrid")
sinplot()
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "5A71C42CCE4540528BAE44899F5D53D4", "mdEditEnable": false, "jupyter": {}, "tags": []}

在很多场合下（比如ppt展示时，用户不会详细读数据，而主要看趋势），用户对网格的需求是不大的，此时我们可以去掉网格。

```{code-cell} ipython3
---
id: 607F6258AD95479885E2D71027CA4AA9
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("dark")
sinplot()
```

```{code-cell} ipython3
---
id: 1CC563323C4E47B086CE73F805404C2F
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("white")
sinplot()
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "848C0426A7444B438084EAC53BCD6F55", "mdEditEnable": false, "jupyter": {}, "tags": []}

ticks风格介于grid风格与完全没有grid的风格之间，坐标轴上提供了刻度线。

```{code-cell} ipython3
---
id: 820716E0D24F4D11AA0A7E1EEC6C9CF8
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("ticks")
sinplot()
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "99CEDAE6A5BB45AC8A510639FD1B64A2", "mdEditEnable": false, "jupyter": {}, "tags": []}

### 移除侧边边界线

```{code-cell} ipython3
---
id: 46C32FD48CA140468FD26831D570113E
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sinplot()
sns.despine()
```

+++ {"slideshow": {"slide_type": "slide"}, "id": "7FE14D10070D49068DB7B1F3A5707A7D", "mdEditEnable": false, "jupyter": {}, "tags": []}

当然，左侧和下方的线也是可以移除的。

```{code-cell} ipython3
---
id: DC7E090678BB4DE58961AB000BD8A8F5
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("white")
sns.boxplot(data=data, palette="deep")
sns.despine(left=True,bottom=True)
```

+++ {"id": "8B84A9137D58421B800478F4A2B3F532", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

### 自定义seaborn styles

+++ {"id": "38DEB13278D74F05ABDB3E3FDA3BFCAE", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

当然了，如果这五种seaborn自带风格也不能满足你的需求，你还可以自行设置自己的风格，可以设置的参数有：

```{code-cell} ipython3
---
id: 112C8F4C92904AAE8C52F1881F9AE2F2
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.axes_style()
```

+++ {"id": "92600781261C41908E804815BC776650", "jupyter": {}, "tags": [], "slideshow": {"slide_type": "slide"}, "mdEditEnable": false}

设置的方法如下：

```{code-cell} ipython3
---
id: E993CB44A5CE4CC9A4492FACA0AAE34A
jupyter: {}
slideshow:
  slide_type: slide
tags: []
---
sns.set_style("white", {"ytick.right": True,'axes.grid':False})
sinplot()
```

