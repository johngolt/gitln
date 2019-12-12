<!--
    作者：华校专
    email: huaxz1986@163.com
**  本文档可用于个人学习目的，不得用于商业目的  **
-->
# 类的设计模式
1.Python不会执行同名函数的重载，而只会用新对象覆盖旧对象

```
  class A:
	def func(self,x):
		pass
	def func(self,x,y):
		pass
```
由于`class`、`def`均为可执行代码，因此`func`变量名这里被重新赋值了。

  ![同名函数的覆盖](../imgs/python_27_1.JPG)

2.委托设计：在Python中委托通常以拦截`.__getattr__(self,'name')`来实现。该方法会拦截对不存在属性的读取

* 代理类实例对象可以利用`.__getattr__(self,'name')`将任意的属性读取转发给被包装的对象
* 代理类可以有被包装对象的借口，且自己还可以有其他接口

  ![代理类](../imgs/python_27_2.JPG)

3.Python支持变量名压缩的概念：`class`语句内以`__`（两个下划线）开头但是结尾没有`__`（两个下划线）的变量名（如`__x`)会自动扩张为包含所在类的名称（如`_classname__x`）

* 变量名压缩只发生在`class`语句内，且仅仅针对`__x`这种以`__`开头的变量名
* 该做法常用于避免实例中潜在的变量名冲突

  ![class的变量名压缩](../imgs/python_27_3.JPG)

4.Python3中，实例方法有两种形式：

* 普通函数方法：通过对类名进行点号运算而获得类的函数属性，如`classname.func`，会返回普通函数方法。
	* 若调用的是实例方法，必须明确提供实例对象作为第一个参数，如`classname.func(obj,arg)`
	* 若调用的是一般的方法，则遵守普通函数调用规则即可`classname.func(arg)`
* 绑定方法：通过对实例对象进行点号运算而获得类的函数属性，如`obj.func`，会返回绑定方法对象。Python在绑定方法对象中，自动将实例和函数打包
	* 绑定方法调用时，不需要手动传入实例对象，如`obj.func(arg)`
	* 绑定方法的`__self__`属性引用被绑定的实例对象，`__func__`属性引用该类的该函数对象

  ![普通方法和绑定方法](../imgs/python_27_4.JPG)

5.多重继承：子类可以继承一个以上的超类。超类在`class`语句首行括号内列出，以逗号分隔

* 子类与其实例继承了列出的所有超类的命名空间
* 搜索属性时，Python会从左到右搜索`class`首行中的超类，直到找到匹配的名字

   ![多重继承](../imgs/python_27_5.JPG)

6.工厂函数：通过传入类对象和初始化参数来产生新的实例对象：

```
  def factory(classname,*args,**kwargs):
	return classname(*args,**kwargs)
```

  ![多重继承](../imgs/python_27_6.JPG)

7.抽象超类：类的部分行为未定义，必须由其子类提供

* 若子类也未定义预期的方法，则Python会引发未定义变量名的异常
* 类的编写者也可以用`assert`语句或者`raise`异常来显式提示这是一个抽象类

```
  class A:
	def func(self):
		self.act() #该方法未实现
	def act(self):
		assert False, 'act must be defined!'
  class ChildA(A):
	def act(self):
		print('in ChildA act')
  x=child()
  x.func()
```

这里的核心在于：超类中`self.act()`调用时，`self`指向的有可能是真实的实例对象（子类对象）
	
  ![抽象超类](../imgs/python_27_7.JPG)

	






