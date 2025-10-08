## 基础python运算
### 字典
```python
hyperparameters = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "Adam",
    "epochs": 10
}
                           # 访问值 (通过键)
lr = hyperparameters["learning_rate"] # 0.001
print(f"Learning Rate: {lr}")
                           # 修改值
hyperparameters["epochs"] = 20
print(f"Updated hyperparameters: {hyperparameters}")

                           # 添加新键值对
hyperparameters["loss_function"] = "CrossEntropy"
print(f"Added loss: {hyperparameters}")
                           # 获取所有键或值
print(f"Keys: {hyperparameters.keys()}")
print(f"Values: {hyperparameters.values()}")
```
### 求导/偏导  
利用sympy库进行导数/偏导数运算
```python
import sympy
x = sympy.symbols('x')     # 定义符号变量 x
f = x**2                   # 定义函数 f(x) = x^2
f1=sympy.diff(f, x)        # 计算导数
print("f'(x)=",f1)
re=f1.subs(x, 2)           # 计算在 x=2 处的导数值
print("f'(2)=",re)
```
### 梯度
在降一维的体（面、线）上（降f(x)的那根轴）的方向向量，模为导数的值。
```python
import sympy
x,y=sympy.symbols('x y')
f=x*y+x+y
fx=sympy.diff(f,x)         # 对于x的偏导数
fy=sympy.diff(f,y)         # 对于y的偏导数
a=(fx.subs({x:1,y:1}),fy.subs({x:1,y:1}))
print(a)
```