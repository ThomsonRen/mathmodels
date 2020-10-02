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

# Jupyter notebook 安装教程


我们使用Anaconda 进行Python环境的配置。使用Anaconda第一次安装相对比较耗时，但是在后续学习时会省配置环境的其他麻烦。Anaconda是一个免费开源的Python和R语言的发行版本，用于计算科学（数据科学、机器学习、大数据处理和预测分析），Anaconda致力于简化包管理和部署。Anaconda的包使用软件包管理系统Conda进行管理。超过1200万人使用Anaconda发行版本，并且Anaconda拥有超过1400个适用于Windows、Linux和MacOS的数据科学软件包。


```{figure} ../_static/lecture_specific/anaconda/anaconda-logo.jpg
---
height: 300px
name: anaconda-1
---
```


## Step1安装包下载
下载地址： https://www.anaconda.com/distribution/
根据电脑操作系统选择 Windows 或者 Mac。选择图形安装界面(graphical installer)。
```{figure} ../_static/lecture_specific/anaconda/anaconda-web.jpg
---
height: 300px
name: anaconda-web
---
```


## Step2 安装
打开下载好的软件安装包，进行安装。



## Step3 测试是否安装成功

如果是Windows系统，按win键，输入jupyter，会出现下图界面，单击打开。

```{figure} ../_static/lecture_specific/anaconda/win.jpg
---
height: 300px
name: win
---
```

如果是mac系统，打开terminal， 输入jupyter notebook, 然后单击回车也可以打开。

正确打开后，浏览器会创建一个新页面，如下图所示。
```{figure} ../_static/lecture_specific/anaconda/web-1.jpg
---
height: 300px
name: web-1
---
```


同时，后台也会打开这样一个命令行窗口，在python运行时请不要关闭。
```{figure} ../_static/lecture_specific/anaconda/terminal.jpg
---
height: 300px
name: terminal
---
```


按以下操作，新建一个notebook文件
```{figure} ../_static/lecture_specific/anaconda/new-file.jpg
---
height: 300px
name: new-file
---
```


新建成功后出现以下界面。在其中输入 `print(“hello world”)`，如果输出hello world 即为安装成功, Enjoy! 