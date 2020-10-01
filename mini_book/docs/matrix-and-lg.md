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

+++ {"tags": [], "slideshow": {"slide_type": "slide"}, "id": "8478FBF13EA7419BA3A250ED42FE1112", "mdEditEnable": false, "jupyter": {}}

# 矩阵和线性代数基础回顾

## Matrix and Vectors


```{admonition} Definition of Matrix

If $m$ and $n$ are positive integers, then an $m \times n$ (read "$m$ by $n$") matrix is a rectangular array
```

$$
\left[\begin{array}{rrrrr}
a_{11} & a_{12} & a_{13} & \ldots & a_{1 n} \\
a_{21} & a_{22} & a_{23} & \ldots & a_{2 n} \\
a_{31} & a_{32} & a_{33} & \ldots & a_{3 n} \\
\vdots & \vdots & \vdots & & \vdots \\
a_{m 1} & a_{m 2} & a_{m 3} & \ldots & a_{m n}
\end{array}\right]
$$




in which each **entry** $a_{ij}$ of the matrix is a number. An $m \times n$ matrix has $m$ rows and $n$ columns. Matrices are usually denoted by capital letters.


| Name | Size | Example |
| :------: | :------: | :-----: |
| Row vector | $1 \times n$ | $\begin{bmatrix}3 & 7 & 2 \end{bmatrix}$ |
| Column vector | $n \times 1$ | $\begin{bmatrix}4 \cr 1 \cr 8 \end{bmatrix}$ |
| Square vector | $n \times n$ | $\begin{bmatrix}9 & 13 & 5 \cr 1 & 11 & 7 \cr 2 & 6 & 3 \end{bmatrix}$|


| Diagonal matrix | Identity matrix | Zero matrix |
| :------: | :------: | :------: |
| $A = \begin{bmatrix}\begin{matrix}\lambda_1 & 0 \\ 0 & \lambda_2 \end{matrix} & \text{0} \\ \text{0} & \begin{matrix}\ddots & 0 \\ 0 & \lambda_n \end{matrix} \end{bmatrix}$ <br/><br/> $A = diag(\lambda_1, \lambda_2, \cdots, \lambda_n)$ | $I_n = \begin{bmatrix}\begin{matrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} & \begin{matrix}\cdots & 0 \\ \cdots & 0 \\ \cdots & 0 \end{matrix} \\ \begin{matrix}\vdots & \vdots & \vdots \\ 0 & 0 & 0 \end{matrix} & \begin{matrix}\ddots & 0 \\ 0 & 1 \end{matrix} \end{bmatrix}$ <br/><br/> $I_n = diag(1, 1, \cdots, 1)$ | $O = \underbrace{\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}}_{2 \times 3 \ zero \ matrix}$|



## Operations with Matrices: Addition and Scalar Multiplication


```{admonition} Definition of Matrix Addition

If $A = [a_{ij}]$ and $B = [b_{ij}]$ are matrices of order $m \times n$, then their sum is the $m \times n$ matrix given by 

$$
A + B = [a_{ij} + b_{ij}]
$$

The sum of two matrices of different orders is undefined.  

$$ 
\begin{bmatrix}-1 & 2 \\ 0 & 1 \end{bmatrix} + \begin{bmatrix}1 & 3 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix}-1+1 & 2+3 \\ 0+(-1) & 1+2 \end{bmatrix} = \begin{bmatrix}0 & 5 \\ -1 & 3 \end{bmatrix} 
$$ 

$$ 
A-B=A+(-1)B 
$$
```


```{admonition} Definition of Scalar Multiplication


If $A=[a_{ij}]$ is an $m \times n$ matrix and $c$ is a scalar, then the **scalar multiple** of $A$ by $c$ is the $m \times n$ matrix given by 

$$
cA=[ca_{ij}]
$$ 

$$
A=\begin{bmatrix}2 & 2 & 4 \\ -3 & 0 & -1 \\ 2 & 1 & 2\end{bmatrix} \quad 3A=3\begin{bmatrix}2 & 2 & 4 \\ -3 & 0 & -1 \\ 2 & 1 & 2\end{bmatrix}
$$
```

```{admonition} Properties of Matrix Addition and Scalar Multiplication

Let $A$, $B$, and $C$ be $m \times n$ matrices and let $c$ and $d$ be scalars.  
1. $A+B=B+A$ &emsp;Commutative Property of Matrix Addition
2. $A+(B+C)=(A+B)+C$ &emsp;Associative Property of Matrix Addition
3. $(cd)A=c(dA)$ &emsp;Associative Property of Scalar Multiplication
4. $1A=A$ &emsp;Scalar Identity Property
5. $c(A+B)=cA+cB$ &emsp;Distributive Property
6. $(c+d)A=cA+dA$ &emsp;Distributive Property
```

## Operations with Matrices: Matrix Multiplication

```{admonition} Definition of Matrix Multiplication


If $A=[a_{ij}]$ is an $m \times n$ matrix and $B=[b_{ij}]$ is an $n \times p$ matrix, then the product $AB$ is an $m \times p$ matrix $$AB=[c_{ij}]$$ where $c_{ij}=a_{i1}b_{ij}+a_{i2}b_{2j}+a_{i3}b_{3j}+\cdots+a_{in}b_{nj}$.
``` 


```{figure} ../_static/lecture_specific/fundamental/picture19.jpg
---
height: 300px
name: pic-19
---

```

```{figure} ../_static/lecture_specific/fundamental/picture20.jpg
---
height: 300px
name: pic-20
---

```


```{admonition} Properties of Matrix Multiplication 




Let $A$, $B$, and $C$ be matrices and let $c$ be a scalar.  
1. $A(BC)=(AB)C$ &emsp;Associative Property of Matrix Multiplication
2. $A(B+C)=AB+AC$ &emsp;Distributive Property
3. $(A+B)C=AC+BC$ &emsp;Distributive Property
4. $c(AB)=(cA)B=A(cB)$ &emsp;Associative Property of Scalar Multiplication

```

```{figure} ../_static/lecture_specific/fundamental/picture21.jpg
---
height: 300px
name: pic-21
---

```

## Operations with Matrices: Matrix Transpose

+++

In linear algebra, the **transpose** of a matrix is an operator which flips a matrix over its diagonal, that is it switches the row and column indices of the matrix by producing another matrix denoted as $A^T$ (also written $A'$, $A^{tr}$, $^{t}A$ or $A^t$). It is achieved by any one of the following equivalent actions:
+ reflect $A$ over its main diagonal (which runs from top-left to bottom-right) to obtain $A^T$,
+ write the rows of $A$ as the columns of $A^T$,
+ write the columns of $A$ as the rows of $A^T$.

+++

1. $(A^T)^T=A$
2. $(A+B)^T=A^T+B^T$
3. $(AB)^T=B^TA^T$
4. $(cA)^T=cA^T$

+++

***

+++

## Homework

```{admonition} Homework

$$
A=\begin{bmatrix}1&2&3\\2&3&4\\4&5&6\end{bmatrix}
$$

$$
B=\begin{bmatrix}1&3&1\\2&1&1\\1&1&4\end{bmatrix}
$$

计算：$A+B$, $A-B$, $3A-2B$, $A \cdot B$, $(A+2B)\cdot A$  
要求：不使用计算器，手算得到结果
```
