import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
xs=np.array([1,2,3,4,5])
ys=np.array([1,2,3,4,5])
w=3
l=0.01


def forward(x,w):         # 进步
    return x*w
def loss(xs,ys,w):        # 损失函数
    cost=0                  
    assert len(xs)==len(ys) 
    for x,y in zip(xs,ys):
        cost=(x*w-y)**2+cost
        return np.mean(cost)
def ge(xs,ys,w):          # 梯度下降
    f1=0
    re=0
    w0=sym.symbols('w0')
    for x,y in zip(xs,ys):
        f=(w0*x-y)**2
        f1=sym.diff(f,w0)
        re=re+f1.subs(w0,w)
    return re/len(xs)



plt.plot(xs,ys)
for i in range(500):
    py=w*xs
    co=loss(xs,ys,w)
    w=w-l*ge(xs,ys,w)
    plt.plot(xs,py)
plt.show()