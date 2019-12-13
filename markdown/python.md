动态类型是Python多态的基础，因为没有类型约束。Python的多态是`x.method`的方法运行时，`method`的意义取决于`x`的类型，**<font color='red'>属性总是在运行期解析</font>**

Python允许执行连续比较，且比较链可以任意长：

* `a<b<c`结果等同于`a<b and b<c`
* `a<b>c`结果等同于`a<b and b>c`

Python中的布尔类型为`bool`，它只有两个值`True`和`False`。`True`和`False`是预定义的内置变量名，其在表达式中的行为与整数1和0是一样的。实际上他们就是内置的`int`类型的子类。

3.`set`对象有以下操作：成员关系：`e in x`、差集： `x-y`、并集： `x|y`、交集： `x&y`、对称差集： `x^y`、判定x是否是y的超集： `x>y`、判定x是否是y的子集：`x<y`

3.`set`的方法有：`.add(item)`：向`set`中插入一项，原地修改（返回`None`)。其中`item`为待插入项；`.update(iter)`：求并集，原地修改（返回`None`)。其中`iter`为任何可迭代对象；`.remove(item)`：向`set`中删除一项，原地修改（返回`None`)。其中`item`为待删除项；`.intersection(iter)`：求交集，返回新的`set`对象。其中`iter`为任何可迭代对象。 `set`对象是可迭代的，因此可用于`len()`函数，`for`循环，以及列表解析中，但是因为是无序的所以不支持索引和分片操作。

列表

* `.pop()`方法：删除末尾元素并返回该元素，<font color="red">原地修改</font>，`.pop(index)`方法：删除指定位置元素并返回该元素
* `.remove(val)`：通过值删除元素，**若有多个值，则只删除第一个遇到的值**<font color="red">原地修改</font>
* `.insert(index,val)`：在指定位置插入元素，<font color="red">原地修改</font>
* `.index(val)`：返回指定元素的位置，**若有多个值，则只返回第一个遇到的值所在位置**

元组：`.index(val)`方法：在元组中搜索`val`值所在位置、`.count(val)`方法：在元组中累计`val`值出现的次数

`set`不是序列，它是可变对象，但是元素只能是不可变类型。字典也不是序列，它是可变对象，其元素的值是不限类型，但是键必须是不可变类型。比较操作时，Python能够自动遍历嵌套的对象，从左到右递归比较，要多深有多深。过充中首次发现的差异将决定比较的结果。 



* 字典的迭代：
  * `d.keys()`：返回一个dict_keys对象，它是一个可迭代对象，迭代时返回键序列
  * `d.values()`：返回一个dict_values对象，它是一个可迭代对象，迭代时返回值序列
  * `d.items()`：返回一个dict_items对象，它是一个可迭代对象，
    迭代时返回元组`(键，值)`的序列
* 获取键的值：通过`d.get(key,default_value)`。返回键对应的值，	若键不存在则返回
  `default_value`
* 字典的操作：
  * `d1.update(d2)`：合并两个字典，原地修改`d1`字典
  * `d.pop(key)`： 从字典中删除`key`并返回该元素的值
  * `del d[key]`：从字典中删除`key`但是不返回该元素的值
  * `d[key]=value`：原地的添加/修改字典。当向不存在的键赋值时，相当于添加新元素

* `d.keys()`、`d.values()`、`d.items()`返回的是可迭代对象，他们称为视图对象，
  而不是列表。修改字典，则这些视图对象会相应变化

变量名由：下划线或字母开头，后面接任意字母、数字、下划线

* 以单下划线开头的变量名不会被`from module import *`语句导入，如变量名`_x`
* 前后双下划线的变量名是系统预定义的，对解析器有着特殊的意义，如变量名`__x__`
* 仅前面有双下划线的变量名视为类的本地变量，如变量名`__x`

扩展的序列解包赋值：收集右侧值序列中未赋值的项为一个列表，将该列表赋值给带星号`*`的变量

* 左边的变量名序列长度不需要与值序列的长度相等，其中只能有一个变量名带星号`*`
  * 若带星号`*`变量名只匹配一项，则也是产生一个列表，列表中只有一个元素，如`a,*b="12"`，`b`为`[2]`
  * 若带星号`*`变量名没有匹配项，则也是产生空列表，如`a,*b="1"`，`b`为`[]`
* 带星号`*`的变量名可以出现在变量名序列中的任何位置如`*a,b="1234"`，`a`为`[1,2,3]`
* 匹配过程优先考虑不带星号的变量名，剩下的才匹配带星号的变量名
* 以下情况会引发错误：
  * 左侧变量名序列有两个星号，如`*a,*b="abcd"`
  * 左侧变量名序列无星号但是左右长度不匹配，如`a,b="abcd"`
  * 左侧变量名序列星号的名称不在序列中，如`*a='abcd'`



* `while`和`else`缩进必须一致。
* `else`可选。`else`子句在控制权离开循环且未碰到`break`语句时执行。即在正常离开循环时执行（`break`是非正常离开循环）
* 在`while`子句中可以使用下列语句：
  * `break`：跳出最近所在的循环到循环外部
  * `continute`：跳过本次循环后续部分，直接掉转到下一轮循环起始处
  * `pass`：占位符，什么都不做



* `for`和`else`缩进必须一致。
* `else`可选。`else`子句在控制权离开循环且未碰到`break`语句时执行。即在正常离开循环时执行（`break`是非正常离开循环）
* 在`for`子句中可以使用`break`、`continute`、`pass`语句
* `target_var`是赋值目标，`iter_obj`是任何可迭代对象。每一轮迭代时将迭代获得的值赋值给`target_var`，然后执行`statement1`



* 占位符有多种：  
  `%s`：字符串； `%r`：也是字符串，但用`repr()`得到的而不是`str()`；  

  `%c`：字符； `%d`：十进制整数； `%i`：整数； `%e`：浮点指数；  

  `%f`： 浮点十进制；`%%`：百分号 `%`， `%g`：自动转成浮点`%e`或者`%f`  

* 转换通用目标结构为：`%[(key_name)][flags][width][.precision]type`

  * `key_name`：用于从右边字典中取对应键的值，如：`"%(n)%d %(x)%s" %{"n":3,"x":"apples"}` 
  * `flags`：如果为`-`则左对齐；如果为`+`则为正负号；如果为`0`：则补零
  * `width`： 指定位宽（包括小数点在内），至少为`width`字符宽度
  * `precision`：指定小数点后几位
  * `type`为类型，如`d`,`r`,`f`,`e`等  

11.格式化字符串除了使用字符串格式化表达式之外，还可以通过字符串的`.format()`
  方法达到同样的效果。

* `.format()`方法支持位置参数、关键字参数、以及二者的混合。
  * 位置参数： `"{0},{1},{2}".format('abc','def','ghi')`
  * 关键字参数：`"{k1},{k2},{k3}".format(k1='abc',k2='def',k3='ghi')`
  * 混合使用：`"{0},{1},{k}".format('abc','def',k='ghi')`  
* 格式化字符串中可以指定对象的属性、字典键、序列的索引：
  * 指定字典的键：`"{0[a]}".format({'a':'value'})`，<font color='red'>注意这里的键`a`并没有引号包围</font>
  * 指定对象的属性：`"{0.platform}".format(sys)`，也可以用关键字参数：
    `"{obj.platform}".format(obj=sys)`
  * 指定序列的索引：`"{0[2]}".format("abcd")` ，这里只能进行正索引值，且不能分片 
* 通用目标结构为： `{fieldname!conversionflag:formatspec}`
  * `fieldname`为位置数字 0,1,2,... 或者为关键字，它后面可选地跟随
    * `.name`：则指定对象的属性
    * `[index]`：指定了索引
    * `[key]`：指定了字典的键
  * `conversionflag`为转换标记，可以为：
    * `r`：在该值上调用一次`repr()`函数
    * `s`：在该值上调用一次`str()`函数
    * `a`：在该值上调用一次`ascii()`函数
  * `formatspec`为格式，其结构为：
    `[[fill] align] [sign] [#] [0] [width] [.precision] [type]`
    * `fill`一般与`align`为`=`时配合
    * `align`为对齐：
      * `<`：左对齐
      * `>`：右对齐
      * `=`：必须指定`fill`（单个字符），此时用这个字符填充
      * `^`：居中对齐
    * `sign`：为正负号标记
    * `#`：作用未知
    * `0`：补0
    * `width`：位宽
    * `.precision`：精度
    * `type`：为类型，如`d`,`r`,`f`,`e`等，
      但与格式化字符串表达式相比，多了`b`（二进制格式输出）  
  * 某些值可以从`.format()`的参数中获取，如`"{0:+0{1}d}".format(128,8)`，
    其指定精度信息从`format()`的参数中取(参数8)

13.由于Python内部会暂存并重复使用短字符串来进行优化，因此该短字符串在内存中只有一份。  

​	



##### 类 class

Python中，类`class`与实例`instance`是两种不同的对象类型：类对象是实例对象的工厂；类对象与实例对象都有各自独立的命名空间；实例对象可自动存取类对象中的变量名；类属性为所有的实例对象提供状态和行为，它是由该类创建的所有实例对象共享的

每个实例对象都有自己的命名空间。同一个类的实例对象不一定属性都相同，每一个实例对象继承类的属性并创建了自己的命名空间，类创建的实例对象是有新的命名空间。刚开始该命名空间是空的，但它会继承创建该实例所属类对象的属性。继承的意思是，虽然实例对象的命名空间是空的。但是名字查找会自动上升到类对象的名字空间去查找，可以在`class`语句外创建类对象的新属性，通过向类对象直接赋值来实现。

6.类可以继承。被继承的类称为超类，继承类称为子类。类对象会继承其超类对象中定义的所有类属性名称，类对象的 `.__dict__`属性是类对象的命名空间，是一个类字典对象`mappingproxy`对象 ； 实例对象的 `.__dict__`属性是实例对象的命名空间，是一个字典； 实例对象的`.__class__`属性是它所属的类，类对象的`__bases__`属性是它超类对象的元组，类对象的`__name__`属性是类名，在子类中调用超类的方法：`superClass.func(obj,args)`，其中`obj`通常为`self` 

若子类重新定义了超类的变量名，子类会取代并定制所继承的行为。这称为重载。在Python中，当对象通过点号运算读取属性值时就会发生继承，而且涉及了搜索属性定义树。每次使用`name.attr`时(`name`为实例对象或者类对象），Python会从底部向上搜索命名空间树。先从本对象的命名空间开始，一直搜索到第一个找到的`attr`名字就停止，命名空间树中较低位置处的定义会覆盖较高位置处的定义，继承树的搜索仅仅发生在读取属性值的时候。在写属性值时，执行的是属性的定义（当前命名空间中该名字不存在）或赋值（当前命名空间中该名字已存在）语义。

~~~mermaid
graph BT;
A(实例命名空间)-->B[类命名空间];
B-->C[超类1命名空间];
B-->D[超类2命名空间];
style A fill:#f9f,stroke:#333;
~~~

通过对象持久化来把他们保存在磁盘中。

* `pickle`模块：通用的对象序列化与反序列化工具。它可以将任何对象转换为字节串，以及将该字节串在内存中重建为最初的对象。`pickle`常用接口为：
  * 序列化：
    *  `pickle.dump(obj, file, protocol=None, *, fix_imports=True) `: 将`obj`对象序列化并写入`file`文件对象中
    *  `pickle.dumps(obj, protocol=None, *, fix_imports=True)`：将`obj`对象序列化并返回对应的字节串对象（并不写入文件中） 
  * 反序列化：
    * `pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict") `：从`file`对象中保存的字节串中读取序列化数据，反序列化为对象
    * `pickle.loads(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict")` ：从字节串中读取序列化数据，反序列化为对象

* `shelve`模块：以上两个模块按照键将Python对象存/取到一个文件中。`shelve`模块提供了一个额外的结构层。允许按照键来存储`pickle`处理后的对象

12.`shelve`模块用法：它用`pickle`把对象转换为字节串，并将其存储在一个`dbm`文件的键之下；它通过键获取`pickle`化的字节串，并用`pickle`在内存中重新创建最初的对象

* 一个`shelve`的`pickle`化对象看上去就像字典。`shelve`自动把字典操作映射到存储|读取在文件中的对象

  * 存储的语法：

    ```
    import shelve
     	db=shelve.open('filename') #打开
     	for obj in objList:
    db[obj.name]=obj #写入
    db.close() #关闭
    ```

  * 读取的语法:

    ```
    import shelve
    db=shelve.open('filename') #打开
    for key in db:#像字典一样访问
    print(key,'=>',db[key]) #读取
    db.close() #关闭
    ```

* 载入重建存储的对象时，不必`import`对象所属类。因为Python对一个对象进行`pickle`操作时，记录了`self`实例属性，以及实例所属类的名字和类的位置。当`shelve`获取实例对象并对其进行`unpickle`时，Python会自动重新`import`该类。

3.在Python3中所有的类都是新式类。

* 所有的类都是从`object`内置类派生而来
* `type(obj)`返回对象实例所属的类对象 
* `type(classname)`返回`"type"`，因为所有`class`对象都是`type`的实例
* 由于所有`class`均直接或者间接地派生自`object`类，因此每个实例对象都是`object`类的实例
* `object`是`type`类的实例，但是同时`type`又派生自`object`

Python3中的类有一个`.__slots__`属性，它是一个字符串列表。这个列表限定了类的实例对象的合法属性名。如果给实例赋了一个`.__slots__`列表之外的属性名会引发异常，当有`.__slots__`列表存在时，默认会删除`.__dict__`属性，而`getattr()`，`setattr()`以及`dir()`等函数均使用`.__slots__`属性，因此仍旧可以正常工作

* 在继承中: 若子类继承自一个没有`.__slots__`的超类，则超类的`.__dict__`属性可用，则子类中的`.__slots__`没有意义。因为子类继承了超类的`.__dict__`属性，若子类有`.__slots__`，超类也有`.__slots__`，子类的合法属性名为父类和子类的`.__slots__`列表的并集，若超类有`.__slots__`，子类未定义`.__slots__`，则子类将会有一个`.__dict__`属性

一个添加了语法糖的方案为：

```
  class A:
    	def __init__(self):
        	self._x = None
    	@property #定义了一个property get函数，必选
    	def x(self): # property name 就是 get函数的函数名
        	"""I'm the 'x' property."""
        	return self._x
    	@x.setter #定义了一个property set函数，可选
    	def x(self, value):
        	self._x = value
    	@x.deleter #定义了一个property del函数，可选
    	def x(self):
        	del self._x
```

Python类中有两种特殊的方法：`staticmethod`方法和`classmethod`方法

* `staticmethod`方法：当以实例对象调用`staticmethod`方法时，Python并不会将实例对象传入作为参数；而普通的实例方法，通过实例对象调用时，Python将实例对象作为第一个参数传入	
* `classmethod`方法：当以实例对象或者类对象调用`classmethod`方法时，Python将类对象（如果是实例对象调用，则提取该实例所属的类对象）传入函数的第一个参数`cls`中		

总结一下，类中可以定义四种方法：

* 普通方法：方法就是类对象的一个属性，执行常规函数调用语义`classname.method(args)`
* 实例方法：传入一个实例作为方法的第一个实参。调用时可以：
  * `obj.method(args)`:通过实例调用
  * `classname.method(obj,args)`：通过类调用
* `staticmethod`方法：* `obj.method(args)`通过实例调用时，执行的是`classname.method(args)`语义
* `classmethod`方法：* `obj.method(args)`执行的是`classname.method(classname,args)`语义

7.类的实例方法中，用哪个实例调用的该方法，`self`就是指向那个实例对象，类的`classmethod`方法中，用哪个类调用该方法，`cls`就指向那个类对象。类对象与实例对象都是可变对象，可以给类属性、实例属性进行赋值，这就是原地修改。这种行为会影响对它的多处引用。若类的某个属性是可变对象，则对它的修改会立即影响所有的实例对象。多重继承中，超类在`class`语句首行内的顺序很重要。Python搜索继承树时总是根据超类的顺序，从左到右搜索超类。类对象的`.__mro__`属性。它是一个`tuple`，里面存放的是类的实例方法名解析时需要查找的类。Python根据该元组中类的前后顺序进行查找。类对象的`.__mro__`列出了`getattr()`函数以及`super()`函数对实例方法名字解析时的类查找顺序。

12.`super()`函数：`super()`返回一个`super`实例对象，它用于代理实例方法/类方法的执行

* `super(class,an_object)`：要求`isinstance(an_object,class)`为真。代理执行了实例方法调用
* `super(class,class2)`：要求 `issubclass(class2,class)`为真。代理执行了类方法调用

有两种特殊用法：`super(class)`：返回一个非绑定的`super`对象，在类的实例方法中，直接调用`super()`，等价于`super(classname,self)`，这里`self`可能是`classname`子类实例，在类的类方法中，直接调用`super()`，等价于`super(classname,cls)`（这里`cls`可能是`classname`子类）

原理：`super`的原理类似于：

``` 
def super(cls,instance):
	mro=instance.__class__.__mro__ #通过 instance生成 mro
	return mro[mro.index(cls)+1] #查找cls在当前mro中的index,返回cls的下一个元素
```

示例：

```
class Root:
	def method1(self):
		print("this is Root")
class B(Root):
	def method1(self):
		print("enter B")
		print(self)
		super(B,self).method1() #也可以简写为 super().method1()
		print("leave B")
class C(Root):
	def method1(self):
		print("enter C")
		print(self)
		super().method1() #也可以写成super(C,self).method1()
		print("leave C")
class D(B,C):
	pass
```

* 调用`D().method1()`--> `D`中没有`method1` 
* `B`中找到（查找规则：`D.__mro__`)  --> 执行`B`中的`method1`。此时`self`为D实例。`D.__mro__`中，`B`的下一个是`C`，因此`super(B,self）.method1()`从类`C`中查找`method1`。
* 执行`C`的`method1`。此时`self`为D实例。`D.__mro__`中，`C`的下一个是`Root`，因此`super(C,self）.method1()`从类`Root`中查找`method1`。
* 执行`Root`的`method1`。
* `print(self)`可以看到，这里的`self`全部为 `D`的实例

##### 元类

所有用户定义的类都是`type`类对象的实例，`type`类是应用最广的元类。`class`语句的内部机制：在一条`class`语句的末尾，Python会调用`type`类的构造函数来创建一个`class`对象。

```python
MyClass=type(classname,superclasses,attributedict) #新建了一个类，类名叫MyClass
# classname:类名，会成为MyClass类的 .__name__属性
# superclasses:类的超类元组，会成为MyClass类的 .__bases__属性
# attributedict:类的命名空间字典，会成为MyClass类的 .__dict__ 属性
```

* `type`类定义了一个`.__call__(...)`方法。该方法运行`type`类定义的两个其他方法：
  * `.__new__(mclass,classname,superclasses,attributedict)`方法，它返回新建的`MyClass`类，`mclass`：为本元类，这里是`type`类，`classname`：为被创建的类的类名，这里是`'MyClass'`，`superclasses`：为被创建的类的超类元组，`attributedict`：为被创建的类的名字空间字典  
  * `.__init__(customclass,classname,superclasses,attributedict)`方法，它初始化新建的`MyClass`类，`customclass`：为被创建的类，这里是`MyClass`类，`classname`：为被创建的类的类名，这里是`'MyClass'`，`superclasses`：为被创建的类的超类元组，`attributedict`：为被创建的类的名字空间字典  

5.所有的类型均由`type`类创建。要通知Python用一个定制的元类来创建类，可以直接声明一个元类来拦截常规的类创建过程。所有元类必须是`type`的子类

```python
class MetaClass(type):
	def __new__(mclass,classname,superclasses,attributedict):		
		return type.__new__(mclass,classname,superclasses,attributedict)
	def __init__(customclass,classname,superclasses,attributedict):
		return type.__init__(customclass,classname,superclasses,attributedict)
class MyClass(metaclass=MetaClass):
	pass
```

* 继承的超类也列在括号中，但是要在元类之前，也用逗号分隔：
  `class MyClass(BaseCls1,BaseCls2,metaclass=MetaClass)`
* 使用元类声明后，在`class`语句底部进行创建`MyClass`类时，改为调用元类`MetaClass`而不是默认的`type`：`MyClass=Meta('MyClass',superclasses,attributedict)`
* 元类`MetaClass`要实现元类协议：
  * 重载元类的`.__new__(Meta,classname,superclasses,attributedict)`方法，它返回新建的`MyClass`类
  * 重载元类的`.__init__(customclass,classname,superclasses,attributedict)`方法，
    它初始化新建的`MyClass`类，`type`类的`.__call__(...)`方法将创建和初始化`MyClass`类对象的调用委托给元类``MetaClass`

7.事实上元类只用于创建类对象，元类并不产生元类自己的实例。因此元类的名字查找规则有些不同：`.__call__`，`.__new__`，`.__init__`方法均在类中查找

8.元类的继承：元类声明由子类继承，即子类的构建也是由父类的元类负责，如果元类是以函数的方式声明，则子类的构建不再继承这个函数式元类

元类中的属性并不进入自定义类的命名空间，即元类中声明的一些类属性与被创建类的名字空间无关，自定义的类，如果没有显示指定元类，也没有指定父类，则默认使用`type`作为元类

##### 装饰器

1.装饰器是用于包装其他可调用对象的一个可调用对象，<font color='red'>它是一个可调用对象，其调用参数为另一个可调用对象，它返回一个可调用对象</font>：一个函数对象是可调用对象。一个类对象是可调用对象，对它调用的结果就是返回类的实例。实现了`.__call__()`方法的类，其实例对象是可调用对象，对它调用的结果就是调用`.__call__()`方法

装饰器有两种使用形式：函数的装饰器：在函数对象定义的时候使用装饰器，用于管理该函数对象；类的装饰器：在类定义的时候使用该装饰器，用于管理该类以及类的实例

2.函数的装饰器：用于管理函数。函数的装饰器声明为：

```python
@decorator
def func(*pargs,**kwargs):
	pass
func=decorator(func) # 等价于
```

* 执行了装饰器的`def`之后，函数名指向的不再是原来的函数对象，而是：一个可调用对象， 当`decorator`是个函数时由`decorator(func)`函数返回的；`decorator`类的实例，当`decorator`是个类时，由`decorator(func)`构造方法返回

3.类的装饰器：用于管理类。类的装饰器声明为：

```python
@decorator
class A:
	pass
A=decorator(A)
```

* 类的装饰器并不是拦截创建实例的函数调用，而是返回一个不同的可调用对象
* 执行了装饰器的`class`之后，类名指向的不再是原来的类对象，而是：一个可调用对象， 当`decorator`是个函数时由`decorator(func)`函数返回的；`decorator`类的实例，当`decorator`是个类时，由`decorator(func)`构造方法返回

```python
def decorator(func): #定义了一个叫decorator的装饰器
	#某些处理
	return func #返回可调用对象
class decorator: #也可以用类来实现装饰器
	def __init__(self,func):
		self.func=func
	def __call__(self,*args,**kwargs):
		return self.func
def decorator(func): #定义了一个叫decorator的装饰器
	def wrapper(*args):
		#使用func或其他的一些工作
	return wrapper #返回可调用对象
```

4.装饰器的嵌套：

```python
@decoratorA
@decoratorB
@decoratorC
def func():
	pass
f=A(B(C(f)))
```

5.装饰器可以携带参数。函数定义的装饰器带参数：它其实是一个嵌套函数。外层函数的参数为装饰器参数，返回一个函数。内层函数的参数为`func`，返回一个可调用参数，<font color='red'>内层函数才是真正的装饰器</font>

```
def decorator(*args,**kwargs): 
	print("this is decorator1:",args,kwargs)
	def actualDecorator(func): # 这才是真实的装饰器
		...
		return func
	return actualDecorator
```

* 类定义的装饰器带参数：它其实是一个嵌套类。外层类的初始化函数的参数为装饰器参数，外层类的`__call__`函数的参数为`func`，返回值为一个类的实例；内层类的初始化函数参数为`func`；内层类的`__call__`函数使用`func`，<font color='red'>内层类才是真正的装饰器</font>

```
class decorator2:
	class ActualDecorator: #这才是真实的装饰器
		def __init__(self,func):
			...
			self.func=func#记住func
		def __call__(self,*args,**kwargs):
			...
			return self.func(*args,**kwargs) #使用func
	def __init__(self,*args,**kwargs):
		...
	def __call__(self,func):
		...
		return decorator2.ActualDecorator(func) 
```

总结：不带参数的装饰器`decorator`装饰一个名字`F`（可能为函数名、也可能为类名）`@decorator`：则执行的是：`F=decorator(F)`，直接使用`F`；带参数的装饰器`decorator`装饰一个名字`F`（可能为函数名、也可能为类名）`@decorator(args)`：则执行的是：`F=decorator(args)(F)`，间接使用`F` 

##### 管理属性

1.管理属性的工具

* `.__getattr__(self,name)`方法：拦截所有未定义属性的读取（它要么返回一个值，要么抛出`AttributeError`异常；`.__setattr__(self,name,value)`方法：拦截所有属性的读取赋值（包括未定义的、已定义的）
* `.__getattribute__(self,name)`方法：拦截所有属性的读取（包括未定义的、已定义的）
* `property`特性：将特定属性访问定位到`get`方法和`set`方法
* 描述符协议：将特定属性访问定位到具有任意`get`和`set`方法的实例对象

2.`property`：每个`property`管理一个单一的、特定的属性。用法为：

```
class A:	
	def fget(...):
		pass
	def fset(...):
		pass
	def fdel(...):
		pass
	attribute=property(fget,fset,fdel,"doc")  #必须在fget,fset,fdel之后定义
a=A()
a.attribute #调用的是property特性
```

* `property()`函数返回的是一个`property`对象
* 子类继承了超类的`property`，就和类的普通属性一样

3.描述符：描述符是作为独立的类创建，它的实例是赋值给了类属性

* 描述符的实例可以由子类继承
* 描述符的实例管理一个单一的特定的属性
* 从技术上讲，`property()`创建的是一个描述符实例（`property`实例）
* 描述符实例针对想要拦截的属性名访问操作，它提供了特定的方法

描述符类的接口为（即描述符协议）：

```python
class Descriptor:
	def __get__(self,instance,owner):
		pass
	def __set__(self,instance,value):
		pass
	def __delete__(self,instance):
		pass
class A:
	attr=Descriptor()
	...
```

* `instance`参数为：
  * `None`：当用于类的属性访问时（如`cls.attr`）
  * 类`A`的实例对象：当用于实例的属性访问时（如`instance.attr`）
* `owner`参数为：使用该描述符的类`A`
* 当访问类实例或者类属性时，自动调用该类的描述符实例的方法。如果该类的描述符中某些方法空缺则：
  * 若` __set__(self,instance,value)`未定义，则写该属性抛出`AttributeError`，该属性只读
  * 若` __get__(self,instance,owner)`未定义，则读该属性返回一个`Descriptor`实例，
    因为从继承树中可知，该属性返回由类的`attr`变量名指定的对象
* 状态信息可以保持在实例对象中，也可以保存在描述符实例中。因为在这3个方法中，`self`,`instance`都可以访问

4.`.__delattr__(self,name)`方法拦截属性的删除

5.由于`.__getattribute__(self,name)`方法和`.__setattr__(self,name,value)`方法对所有的属性拦截，因此他们的实现特别要小心，注意不要触发无穷递归。

* `.__getattribute__(self,name)`方法中，若要取属性则可以用超类的`.__getattribute__(self,name)`获取。如果通过`.__dict__`方法获取则会再次触发`.__getattribute__(self,name)`的调用，因为`__dict__`本身就是对象的属性。
* `.__setattr__(self,name,value)`方法中，若要设置属性可以用`self.__dict__[name]=value`的方法，或者用超类的`.__setattr__(self,name,value)`方法
* 所有使用内置操作隐式的获取方法名属性，`.__getattr__(self,name)`、`.__setattr__(self,name,value)`、
  `.__getattribute__(self,name)`方法都不会拦截，因为Python在类中查找这样的属性，完全忽略了在实例中查找

7.属性拦截优先级：

* 在读取属性方面，`__getattribute__`优先级最高；在写属性方面，`__setattr__`优先级最高；在删除属性方面，
  `__del__`优先级最高

* 如果没有`__getattribute__`，`__setattr__`与`__del__`，则读写删属性取决于描述符（`property`也是一种特殊的描述符）。其中如果同一个属性指定了多个描述符，则后面的描述符覆盖前面的描述符

* `__getattribute__`与`__getattr__`区别：`__getattribute__`在任何属性读取的时候拦截，而`__getattr__`只有在未定义属性读取的时候拦截（约定俗成地，它要么返回一个值，要么返回`AttributeError`）。其中若二者同时存在则`__getattribute__`优先级较高

实例方法：定义：第一个参数必须是实例对象，该参数名一般约定为“self”，通过它来传递实例的属性和方法；

类方法：定义：使用装饰器`@classmethod`。第一个参数必须是当前类对象，该参数名一般约定为“cls”，通过它来传递类的属性和方法；

静态方法：定义：使用装饰器`staticmethod`。参数随意，没有“self”和“cls”参数，但是方法体中不能使用类或实例的任何属性和方法；静态方法是类中的函数，不需要实例。静态方法主要是用来存放逻辑性的代码，逻辑上属于类，但是和类本身没有关系，也就是说在静态方法中，不会涉及到类中的属性和方法的操作。可以理解为，静态方法是个独立的、单纯的函数，它仅仅托管于某个类的名称空间中，便于使用和维护。



​    描述符的作用是用来代理一个类的属性，需要注意的是描述符不能定义在被使用类的构造函数中，只能定义为类的属性，它只属于类的，不属于实例。类的`__dict__`属性是类的一个内置属性，类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类`__dict__`里。在输出描述符的变量时，会调用描述符中的`__get__`方法，在设置描述符变量时，会调用描述符中的`__set__`方法。

 描述符分为数据描述符和非数据描述符。至少实现了内置`__set__()`和`__get__()`方法的描述符称为数据描述符；实现了除`__set__()`()以外的方法的描述符称为非数据描述符。描述符的优先级的高低顺序：类属性 > 数据描述符 > 实例属性 > 非数据描述符 > 找不到的属性触发`__getattr__()`。

函数在使用@property 装饰器装饰后，会变成类属性，而且会有一个 setter 方法，该方法也是一个装饰器，作用是装饰同属性(特性)的 set 函数。被装饰的函数必须与属性（被@property 装饰器装饰的函数）同名。当类函数被 @property 装饰时，实际上，这个函数已经成为了该类的特性，也就是该类的类属性了，这个过程在解释器导入该模块时就已经确定了。

 

##### 类的设计模式

2.委托设计：在Python中委托通常以拦截`.__getattr__(self,'name')`来实现。该方法会拦截对不存在属性的读取

* 代理类实例对象可以利用`.__getattr__(self,'name')`将任意的属性读取转发给被包装的对象
* 代理类可以有被包装对象的借口，且自己还可以有其他接口

3.Python支持变量名压缩的概念：`class`语句内以`__`（两个下划线）开头但是结尾没有`__`（两个下划线）的变量名（如`__x`)会自动扩张为包含所在类的名称（如`_classname__x`）

* 变量名压缩只发生在`class`语句内，且仅仅针对`__x`这种以`__`开头的变量名
* 该做法常用于避免实例中潜在的变量名冲突

5.多重继承：子类可以继承一个以上的超类。超类在`class`语句首行括号内列出，以逗号分隔

* 子类与其实例继承了列出的所有超类的命名空间
* 搜索属性时，Python会从左到右搜索`class`首行中的超类，直到找到匹配的名字

6.工厂函数：通过传入类对象和初始化参数来产生新的实例对象：

```
  def factory(classname,*args,**kwargs):
	return classname(*args,**kwargs)
```

##### 运算符重载

Python中所有可以被重载的方法名称前、后均有两个下划线字符，以便将它与其他类内定义的名字区分开来，如`__add__`；若使用未定义运算符重载方法，则它可能继承自超类。若超类中也没有则说明你的类不支持该运算，强势使用该运算符则抛出异常  

4.`.__init__(self,args)`方法：称为构造函数。当新的实例对象构造时，会调用`.__init__(self,args)`方法。它用于初始化实例的状态  

5.`.__getitem__(self,index)`和`.__setitem(self,index,value)__`方法：

* 对于实例对象的索引运算，会自动调用`.__getitem__(self,index)`方法，将实例对象作为第一个参数传递，方括号内的索引值传递给第二个参数

* 对于分片表达式也调用`.__getitem__(self,index)`方法。实际上分片边界如`[2:4]`绑定到了一个`slice`分片对象上，该对象传递给了`.__getitem__`方法。  对于带有一个`.__getitem__`方法的类，该方法必须既能针对基本索引（一个整数），又能针对分片调用（一个`slice`对象作为参数；`.__getitem__(self,index)`也是Python的重载迭代方式之一。一旦定义了这个方法，`for`循环每一次循环时可以调用`.__getitem__(self,index)`方法。因此任何响应了索引运算的内置或者用户自定义的实例对象通用可以响应迭代。
* `.__setitem(self,index,value)__`方法类似地拦截索引赋值和分片赋值。第一个参数为实例对象，第二个参数为基本索引或者分片对象，第三个参数为值  

6.`.__index__(self)`方法：该方法将实例对象转换为整数值。即当要求整数值的地方出现了实例对象时自行调用。  

Python的所有迭代环境都会首先尝试调用`.__iter__(self)`方法，再尝试调用`.__getitem__(self,index)`方法。要让实例对象支持多个迭代器，`.__iter__(self)`方法必须创建并返回新的迭代器对象。

* `.__iter__(self)`方法必须返回一个迭代器对象。Python的迭代环境通过重复调用这个迭代器对象的`.__next__(self)`方法，直到发生了`StopIteration`异常
  * `.__iter__(self)`返回的迭代器对象会在调用`.__next__(self)`
    的过程中明确保留状态信息，因此比`.__getitem__(self,index)`方法具有更好的通用性
  * 迭代器对象没有重载索引表达式，因此不支持随机的索引运算
  * `.__iter__(self)`返回的迭代器只能顺序迭代一次。
    因此每次要进行新的一轮循环时必须创建一个新的迭代器对象
* 对于调用`.__getitem__(self,index)`的环境，Python的迭代环境通过重复调用该方法，其中`index`每轮迭代中从 0 依次递增，直到发生了`IndexError`异常  

类通常把`in`成员关系运算符实现为一个迭代，用`.__iter__(self)`方法或`.__getitem__(self,index)`方法。也能实现`.__contains__(self,value)`方法来实现特定成员关系。

* `.__contains__(self,value)`方法优先于`.__iter__(self)`方法，`.__iter__(self)`方法优先于`.__getitem__(self,index)`方法采纳  

`.__getattr__(self,'name')`方法：拦截属性点号运算`obj.name`。只有当对未定义（即不存在）的属性名称进行点号运算时，实例对象会调用此方法，当Python可以从继承树中找到该属性名时，并不会调用`.__getattr__(self,'name')`方法，属性不仅仅是变量名，也可以是方法名

* 内置的`getattr(obj,'name')`函数等价于调用`obj.name`，它执行继承搜索。搜不到时调用`.__getattr__(self,“name”)`方法  
* 如果没有定义`.__getattr__(self,“name”)`方法，则对于不知道如何处理的属性（即找不到的），则Python抛出内置的`AttributeError`异常  

12.`.__setattr__(self,'name',value)`方法：拦截所有的属性赋值语句（无论该属性名是否存在）对于属性赋值语句，因为如果该属性曾经不存在，则一旦赋值就增加了一个新的属性，属性不仅仅是变量名，也可以是方法名，<font color='red'>注意：`.__setattr__(self,'name',value)`方法的函数体内，任何对`self`属性赋值语句(`self.name=value`)都会再次递归调用`.__setattr__(self,'name',value)`函数</font>，为了防止`.__setattr__(self,'name',value)`函数体内的无穷递归，在该方法内的`self`属性赋值要采用属性字典索引的方法：`self.__dict__['name']=value`，内置的`setattr(obj,'name',value)`函数等价于调用`obj.name=value`

13.`.__getattribute__(self,'name')`方法：拦截所有的属性读取，而不仅仅是那些未定义的。 <font color='red'>注意：`.__getattribute__(self,'name')`方法的函数体内，任何对`self`属性读取语句(`self.name`)都会再次递归调用`.__getattribute__(self,'name')`函数。尽量不要重载`.__getattribute__(self,'name')`方法避免无穷递归</font>

14.通过`.__getattr__`与`.__setattr__`方法混合使用可以模拟实例对象的私有属性：

* 实例对象保存一个`self.private`变量名列表
* 对`.__setattr__`与`.__getattr__`，判断属性名是否在`self.private`变量名列表中。
  若是，则抛出异常

> 对于通过`obj.__dict__['name']`访问，可以绕过这种机制

15.`.__add__(self,value)`方法：当实例对象在加法中时调用

17.加法有三种：

* 常规加法：实例对象在`+`左侧，由`.__add__(self,value)`拦截
* 右侧加法：实例对象在`+`右侧，由`.__radd__(self,value)`拦截
* 原地加法：实例对西在`+=`左侧，由`.__iadd__(self,value)`拦截

要实现满足交换律的运算符，要同时重载`.__add__(self,value)`与`.__radd__(self,value)`方法。当不同类的实例对象混合出现在`+`两侧时，Python优先选择左侧的那个类来拦截`+`

* 原地`+=`优先采用`.__iadd__(self,value)`，如果它没有重载，则采用`.__add__(self,value)`

18.每个二元运算符都有类似`+`的右侧和原地重载方法。他们以相似的方式工作。

* 右侧方法通常只有在需要满足交换律时用得到，一般较少使用
* 在实现这些方法时，函数体内注意不要出现递归调用

19.`.__call__(self,*pargs,**kwargs)`方法：函数调用方法。当调用实例对象时，由`.__call__(self,*pargs,**kwargs)`方法拦截。

* `.__call__(self,*pargs,**kwargs)`方法支持所有的参数传递方式 

20.实例对象可以拦截6种比较运算符：`< > <= >= == !=`，对应于

```
	.__lt__(self, other) # <
	.__le__(self, other) # <=
	.__gt__(self, other) # >
	.__ge__(self, other) # >=
	.__eq__(self, other) # ==
	.__ne__(self, other) # !=
```

* 比较运算符全部是左端形式，无右端形式：`3<=obj`会转换成`obj>=3`
* 比较运算符并没有隐式关系。`==`为真，并不意味着`!=`为假。
  因此`.__eq__(self, other)`与`.__ne__(self, other)`必须同时实现而且语义要一致。  

21.在布尔环境中，Python会首先尝试`.__bool__(self)`方法来获取一个直接的布尔值。如果没有这个方法，则 尝试`.__len__(self)`方法根据其结果确定实例对象的真值（非0则为真，0为假）



##### 迭代器和生成器

1.可迭代对象：在逻辑上它保存了一个序列，在迭代环境中依次返回序列中的一个元素值。

2.迭代协议：`.__next__()`方法。

* 任何对象只要实现了迭代协议，则它就是一个迭代器对象
* 迭代器对象调用`.__next__()`方法，会得到下一个迭代结果
* 在一系列迭代之后到达迭代器尾部，若再次调用`.__next__()`方法，则会触发`StopIteration`异常
* 迭代器在Python中是用C语言的速度运行的，因此速度最快  

4.内置的`iter()`函数用于从序列、字典、`set`以及其他可迭代对象中获取迭代器。

* 对任何迭代器对象`iterator`，调用`iter(iterator)`返回它本身
* 迭代器对象实现了迭代协议
* 文件对象本身是一个迭代器对象。即文件对象实现了迭代协议，因此打开多个文件会返回同一个文件对象
* 列表、元组、字典、`set`、字符串等不适迭代器对象，他们没有实现迭代协议。因此每次调用`iter()`均返回一个新迭代器对象。他们支持安装多个迭代器，每个迭代器状态不同
  * 在原地修改列表、`set`、字典时，会实时反映到它们的迭代器上

5.文件迭代器：文件对象本身是一个迭代器（这里文件对象要是读打开）。它的`.__next__()`方法每次调用时，返回文件中的下一行。当到达文件末尾时，`.__next__()`方法会引发`StopIteration`异常。

9.常见的迭代函数：

* `map(func,iterable)`：它将函数`func`应用于传入的迭代器的每个迭代返回元素，返回一个新的迭代器，函数执行结果作为新迭代器的迭代值 

* `zip(iterable1,iterable2,...)`：它组合可迭代对象`iterable1`、`iterable2`、`...`中的各项，返回一个新的迭代器。新迭代器长度由`iterable1`、`iterable2`、`...`最短的那个决定。  
* `enumerate(iterable,start)`：返回一个迭代器对象，它迭代结果是每次迭代返回一个`(index,value)`元组  
* `filter(func,iterable)`：返回一个迭代器对象，它的迭代结果得到`iterable`中部分元素，其中这些元素使得`func()`函数返回为真  
* `reduce(func,iterable,initial)`：对`iterable`中每一项成对地运行`func`，返回最终值 `reduce`函数位于`functools`包内

* `sorted(iterable,key=None,reverse=False)`：排序并返回排好的新列表 

`range`对象不支持`.__next__()`，因此它本身不是迭代器，而`map`、`zip`、`filter`对象都是迭代器。字典的视图：键视图、值视图、字典视图都没有`.__next__()`方法，因此他们都不是迭代器  

14.生成器函数：编写为常规的`def`语句，但是用`yield`语句一次返回一个结果。每次使用生成器函数时会继续上一轮的状态。生成器函数会保存上次执行的状态

* 生成器函数执行时，得到一个生成器对象，它`yield`一个值，而不是返回一个值。	
  * 生成器对象自动实现迭代协议，它有一个`.__next__()`方法
  * 对生成器对象调用`.__next__()`方法会继续生成器函数的运行到下一个`yield`
     	  结果或引发一个`StopIteration`异常
* `yield`语句会挂起生成器函数并向调用者发送一个值。当下一轮继续时，函数会在上一个`yield`表达式返回后继续执行，其本地变量根据上一轮保持的状态继续使用 

16.生成器对象有一个`.send(arg)`方法。该方法会将`arg`参数发送给生成器作为`yield`表达式的返回值，同时生成器会触发生成动作(相当于调用了一次`.__next__()`方法。`yield`表达式的返回值和生成值是不同的。返回值是用于生成器函数内部，`yield`表达式默认返回值为`None`；而生成值是用于生成器函数外部的迭代返回。  

* 生成器对象必须先启动。启动意味着它第一次运行到`yield`之前挂起    要想启动生成器，可以直接使用`next(generatorable)`函数，也可以使用`generatorable.send(None)`方法，或者
  `generatorable.__next__()`方法，`generatorable.send(None)`方法会在传递`yield`表达式的值（默认为`None`返回值），下一轮迭代从`yield`表达式返回开始。每一轮挂起时，`yield`表达式 yield 一个数，但是并没有返回（挂起了该`yield`表达式）

18.生成器函数可以有`return`，它可以出现在函数内任何地方。生成器函数内遇到`return`则触发`StopIteration`异常，同时`return`的值作为异常说明 。可以调用生成器对象的`.close()`方法强制关闭它。这样再次给它`send()`任何信息，都会抛出`StopIteration`异常，表明没有什么可以生成的了  

* `yield from`可以将一个大的生成器切分成小生成器：

* `yield from`能实现代理生成器：

  ```python
  def generator():
      inner_gen=generator2()
      yield from inner_gen #为了便于说明，这里分两行写
  gen=generator()
  ```

  * 对`inner_gen`迭代产生的每个值都直接作为`gen` yield值
  * 所有`gen.send(val)`发送到`gen`的值`val`都会被直接传递给`inner_gen`。
  * `inner_gen`抛出异常：如果`inner_gen`产生了`StopIteration`异常，则`gen`会继续执行`yield from`之后的语句；如果对`inner_gen`产生了非`StopIteration`异常，则传导至`gen`中，导致`gen`在执行`yield from`的时候抛出异常
  * `gen`抛出异常：如果`gen`产生了除`GeneratorExit`以外的异常，则该异常直接 throw 到`inner_gen`中；如果`gen`产生了`GeneratorExit`异常，或者`gen`的`.close()`方法被调用，
    则`inner_gen`的`.close()`方法被调用。
  * `gen`中`yield from`表达式求职结果是`inner_gen`迭代结束时抛出的`StopIteration`异常的第一个参数
  * `inner_gen`中的`return xxx`语句实际上会抛出一个`StopIteration(xxx)`异常，
    所以`inner_gen`中的`return`值会成为`gen`中的`yield from`表达式的返回值。

##### Python

每个对象至少包含三个数据：引用计数、类型、值

 引用计数用于内存管理。要深入了解Python内存管理的内核；类型在`CPython`层使用，用于确保运行时的类型安全性。最后，值，即与对象关联的实际值。`id()`返回对象的内存地址。`is `当且仅当两个对象具有相同的内存地址时才返回True。

###### C中变量

```c
int x=2337	
```

这一行代码在执行时有几个不同的步骤：为整数分配足够的内存；将值分配2337给该内存位置；指示x指向该值；

 以简化的内存视图显示，它可能如下所示： 

![](D:/学习/MarkDown/picture/work/37.png)

另一种思考这个概念的方法是在所有权方面。从某种意义上说，`x`拥有内存位置。首先，`x`恰好是一个可以存储整数的空盒子，可以用来存储整数值。当您给`x`赋值时，您将向`x`拥有的盒子中放入一个值。如果你想引入一个新的变量`y`，

```c
int y=x
```

![](D:/学习/MarkDown/picture/work/38.png)

###### python中的名称

``` python
x=2337
```

与`C`类似，python在执行过程中分解为几个不同的步骤：创建一个`PyObject`；将`PyObject`的`typecode`设置为整数`PyObject`；将`PyObject`的值设置为2337；创建一个名称`x`；将`x`指向新的`PyObject`；将`PyObject`引用计数增加`1`

![](D:/学习/MarkDown/picture/work/39.png)

```python
x=2338
```

这行代码：创建一个新的`PyObject`；将`PyObject`的`typecode`设置为整数；将`PyObject`的值设置为2338；将`x`指向新的`PyObject`；将新的`PyObject`引用计数增加1；将旧的`PyObject`引用计数减少1

![](D:/学习/MarkDown/picture/work/40.png)

```python
y=x
```

![](D:/学习/MarkDown/picture/work/41.png)

###### python中的预实现对象

Python在内存中预先创建了某个对象子集，并将它们保存在全局命名空间中以供日常使用。哪些对象依赖于Python的预实现。`CPython 3.7`预实现对象如下：-5到256之间的整数；仅包含ASCII字母，数字或下划线的字符串。这背后的原因是这些变量很可能在许多程序中使用。通过预先实现些对象，Python可以防止对一致使用的对象进行内存分配调用。

You can also use `map` with more than one iterable. For example, if you want to calculate the mean squared error of a simple linear function `f(x) = ax + b` with the true label `labels`, these two methods are equivalent

```python
diffs = map(lambda x, y: (a * x + b - y) ** 2, xs, labels)
# If we want to replace the element at an index with multiple elements
elems[1:2] = [20, 30, 40]
#If we want to insert 3 values between element at index 0 and element at index 1
elems[1:1]=[20,30,40]
'''If we have nested lists, we can recursively flatten it. That's another beauty of lambda functions -- we can use it in the same line as its creation.'''
nested_lists = [[1, 2], [[3, 4], [5, 6], [[7, 8], [9, 10], [[11, [12, 13]]]]]]
flatten = lambda x: [y for l in x for y in flatten(l)] 
if type(x) is list else [x]
flatten(nested_lists)
```

Note that objects returned by `map` and `filter` are iterators, which means that their values aren't stored but generated as needed. After you've called `sum(diffs)`, `diffs` becomes empty. If you want to keep all elements in `diffs`, convert it to a list using `list(diffs)`. `filter(fn, iterable)` works the same way as `map`, except that `fn` returns a boolean value and `filter` returns all the elements of the `iterable` for which the `fn` returns True. `reduce(fn, iterable, initializer)` is used when we want to iteratively apply an operator to all elements in a list. Lambda functions are meant for one time use. Each time `lambda x: dosomething(x)` is called, the function has to be created, which hurts the performance if you call `lambda x: dosomething(x)` multiple times. When you assign a name to the lambda function as in `fn = lambda x: dosomething(x)`, its performance is slightly slower than the same function defined using `def`, but the difference is negligible.

We'd also like to compare two nodes by comparing their values. To do so, we overload the operator `==` with `__eq__`, `<` with `__lt__`, and `>=` with `__ge__`.  The `locals()` function returns a dictionary containing the variables defined in the local namespace. All attributes of an object are stored in its `__dict__`. Note that manually assigning each of the arguments to an attribute can be quite tiring when the list of the arguments is large. To avoid this, we can directly assign the list of arguments to the object's `__dict__`.  This can be especially convenient when the object is initiated using the catch-all `**kwargs`. 

If we intend that only Encoder, Decoder, and Loss are ever to be imported and used in another module, we should specify that in `parts.py` using the `__all__` keyword.

```python
__all__ = ['Encoder', 'Decoder', 'Loss']
f = lambda *args: sum(args)
g = lambda x,y=3: x+y
```



##### `io`编程

在python中访问文件要先调用一个内置函数`open`，它返回一个与底层文件交互的对象。在处理一个文件时，文件对象使用距离文件开始处的偏移量维护文件中的当前位置。在以只读权限或只写权限打开文件时，初始位置是0；如果以追加权限打开，初始位置是在文件的末尾。`fp = open('sample.txt')`

| 调用方法             | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| `fp.read()`          | 将只读文件剩下的所有内容作为一个字符串返回                   |
| `fp.read(k)`         | 将只读文件中接下来k个字节作为一个字符返回                    |
| `fp.readline()`      | 从文件中读取一行内容，并以此作为一个字符串返回               |
| `fp.readlines()`     | 将文件中的每行内容作为一个字符串存入列表中，并返回该列表     |
| `fp.seek(k)`         | 将当前位置定位到文件的第k个字节                              |
| `for line in fp`     | 遍历文件中每一行                                             |
| `fp.tell()`          | 返回当前位置偏离开始处的字节数                               |
| `fp.write(string)`   | 在只写文件的当前位置将`string`的内容写入                     |
| `fp.writelines(seq)` | 在只写文件的当前位置写入给定序列的每个字符串。除了那些嵌入到字符串中的换行符，这个命令不插入换行符。 |

在磁盘上读写文件的功能都是由操作系统提供的，现代操作系统不允许普通的程序直接操作磁盘，所以，读写文件就是请求操作系统打开一个文件对象，然后，通过操作系统提供的接口从这个文件对象中读取数据，或者把数据写入这个文件对象。

$\text{file-like object}$：像`open()`函数返回的这种有个`read()`方法的对象，在Python中统称为file-like Object。除了`file`外，还可以是内存的字节流，网络流，自定义流等等。$\text{file-like object}$不要求从特定类继承，只要写个`read()`方法就行。`StringIO`就是在内存中创建的$\text{file-like object}$，常用作临时缓冲。

`StringIO`顾名思义就是在内存中读写`str`。`BytesIO`实现了在内存中读写`bytes`，我们创建一个`BytesIO`，然后写入一些`bytes`。

`dumps()`方法返回一个`str`，内容就是标准的`JSON`。类似的，`dump()`方法可以直接把`JSON`写入一个`file-like Object`。要把`JSON`反序列化为Python对象，用`loads()`或者对应的`load()`方法，前者把`JSON`的字符串反序列化，后者从`file-like Object`中读取字符串并反序列化

###### 异步`io`

在一个线程中，CPU执行代码的速度极快，然而，一旦遇到IO操作，就需要等待IO操作完成，才能继续进行下一步操作。这种情况称为同步IO。在IO操作的过程中，当前线程被挂起，而其他需要CPU执行的代码就无法被当前线程执行了。因为一个IO操作就阻塞了当前线程，导致其他代码无法执行，所以我们必须使用多线程或者多进程来并发执行代码，为多个用户服务。每个用户都会分配一个线程，如果遇到IO导致线程被挂起，其他用户的线程不受影响。多线程和多进程的模型虽然解决了并发问题，但是系统不能无上限地增加线程。由于系统切换线程的开销也很大，所以，一旦线程数量过多，CPU的时间就花在线程切换上了，真正运行代码的时间就少了，结果导致性能严重下降。由于我们要解决的问题是CPU高速执行能力和IO设备的龟速严重不匹配，多线程和多进程只是解决这一问题的一种方法。另一种解决IO问题的方法是异步IO。当代码需要执行一个耗时的IO操作时，它只发出IO指令，并不等待IO结果，然后就去执行其他代码了。一段时间后，当IO返回结果时，再通知CPU进行处理。

```python
do_some_code()
f = open('/path/to/file', 'r')
r = f.read() # <== 线程停在此处等待IO操作结果
# IO操作完成后线程才能继续执行:
do_some_code(r)
#异步IO模型需要一个消息循环，在消息循环中，主线程不断地重复“读取消息-处理消息”这一过程
loop = get_event_loop()
while True:
    event = loop.get_event()
    process_event(event)
```

消息模型其实早在应用在桌面应用程序中了。一个GUI程序的主线程就负责不停地读取消息并处理消息。所有的键盘、鼠标等消息都被发送到GUI程序的消息队列中，然后由GUI程序的主线程处理。由于GUI线程处理键盘、鼠标等消息的速度非常快，所以用户感觉不到延迟。某些时候，GUI线程在一个消息处理的过程中遇到问题导致一次消息处理时间过长，此时，用户会感觉到整个GUI程序停止响应了。这种情况说明在消息模型中，处理一个消息必须非常迅速，否则，主线程将无法及时处理消息队列中的其他消息，导致程序看上去停止响应。

消息模型是如何解决同步IO必须等待IO操作这一问题的呢？当遇到IO操作时，代码只负责发出IO请求，不等待IO结果，然后直接结束本轮消息处理，进入下一轮消息处理过程。当IO操作完成后，将收到一条“IO完成”的消息，处理该消息时就可以直接获取IO操作结果。在“发出IO请求”到收到“IO完成”的这段时间里，同步IO模型下，主线程只能挂起，但异步IO模型下，主线程并没有休息，而是在消息循环中继续处理其他消息。这样，在异步IO模型下，一个线程就可以同时处理多个IO请求，并且没有切换线程的操作。对于大多数IO密集型的应用程序，使用异步IO将大大提升系统的多任务处理能力。

###### 协程

子程序，或者称为函数，在所有语言中都是层级调用，比如A调用B，B在执行过程中又调用了C，C执行完毕返回，B执行完毕返回，最后是A执行完毕。所以子程序调用是通过栈实现的，一个线程就是执行一个子程序。子程序调用总是一个入口，一次返回，调用顺序是明确的。

协程的调用和子程序不同。协程看上去也是子程序，但执行过程中，在子程序内部可中断，然后转而执行别的子程序，在适当的时候再返回来接着执行。

##### 线程

线程，有时被称为轻量进程，是程序执行流的最小单元。一个标准的线程由线程ID，当前指令指针(PC），寄存器集合和堆栈组成。线程是进程中的一个实体，是被系统独立调度和分派的基本单位，线程不拥有私有的系统资源，但它可与同属一个进程的其它线程共享进程所拥有的全部资源。一个线程可以创建和撤消另一个线程，同一进程中的多个线程之间可以并发执行。

线程是程序中一个单一的顺序控制流程。进程内有一个相对独立的、可调度的执行单元，是系统独立调度和分派CPU的基本单位指令运行时的程序的调度单位。在单个程序中同时运行多个线程完成不同的工作，称为多线程。Python多线程用于I/O操作密集型的任务。

现代处理器都是多核的，几核处理器只能同时处理几个线程，多线程执行程序看起来是同时进行，实际上是CPU在多个线程之间快速切换执行，这中间就涉及到上下问切换，所谓的上下文切换就是指一个线程Thread被分配的时间片用完了之后，线程的信息被保存起来，CPU执行另外的线程，再到CPU读取线程Thread的信息并继续执行Thread的过程。

###### 线程模块

Python创建Thread对象语法如下：

```
import threadingthreading.Thread(target=None, name=None,  args=())
```

主要参数说明：

- target 是函数名字，需要调用的函数。
- name 设置线程名字。
- `args `函数需要的参数，以元祖( tuple)的形式传入
- Thread对象主要方法说明:
- `run()`: 用以表示线程活动的方法。
- `start()`:启动线程活动。
- `join()`: 等待至线程中止。
- `isAlive()`: 返回线程是否活动的。
- `getName()`: 返回线程名。
- `setName()`: 设置线程名。

Python中实现多线程有两种方式：函数式创建线程和创建线程类。

函数式创建线程：创建线程的时候，只需要传入一个执行函数和函数的参数即可完成`threading.Thread`实例的创建。下面的例子使用Thread类来产生2个子线程，然后启动2个子线程并等待其结束，

```python
import threadingimport time,random,math# idx 循环次数
def printNum(idx):    
    for num in range(idx ):#打印当前运行的线程名字        
        print("{0}\tnum={1}".format(threading.current_thread().getName(), num))
        delay = math.ceil(random.random() * 2)        
        time.sleep(delay)
if __name__ == '__main__':    
    th1 = threading.Thread(target=printNum, args=(2,),name="thread1"  )    
    th2 = threading.Thread(target=printNum, args=(3,),name="thread2" )
    th1.start() #启动2个线程   
    th2.start()
    th1.join()   #等待至线程中止  
    th2.join()    
    print("{0} 线程结束".format(threading.current_thread().getName()))
```

运行脚本默认会启动一个线程，把该线程称为主线程，主线程有可以启动新的线程，Python的threading模块有个current_thread()函数，它将返回当前线程的示例。从当前线程的示例可以获得前运行线程名字，核心代码如下。

```
threading.current_thread().getName()
```

启动一个线程就是把一个函数和参数传入并创建Thread实例，然后调用start()开始执行

创建线程类：直接创建`threading.Thread`的子类来创建一个线程对象,实现多线程。通过继承Thread类，并重写Thread类的run()方法，在run()方法中定义具体要执行的任务。在Thread类中，提供了一个start()方法用于启动新进程，线程启动后会自动调用run()方法。

```python
import threading
import time,random,math
class MutliThread(threading.Thread):    
    def __init__(self, threadName,num):        
        threading.Thread.__init__(self)        
        self.name = threadName        
        self.num = num    
    def run(self):        
        for i in range(self.num):            
            print("{0} i={1}".format(threading.current_thread().getName(), i))             delay = math.ceil(random.random() * 2)            
            time.sleep(delay)
if __name__ == '__main__':    
    thr1 = MutliThread("thread1",3)    
    thr2 = MutliThread("thread2",2)  
    thr1.start()    
    thr2.start()      
    thr1.join()    
    thr2.join()    
    print("{0} 线程结束".format(threading.current_thread().getName()))
```

如果子线程`thread1`和`thread2`不调用join()函数，那么主线程`MainThread`和2个子线程是并行执行任务的，2个子线程加上`join()`函数后，程序就变成顺序执行了。所以子线程用到`join()`的时候，通常都是主线程等到其他多个子线程执行完毕后再继续执行，其他的多个子线程并不需要互相等待。

###### 守护线程

在线程模块中，使用子线程对象用到join()函数，主线程需要依赖子线程执行完毕后才继续执行代码。如果子线程不使用join()函数，主线程和子线程是并行运行的，没有依赖关系，主线程执行了，子线程也在执行。在多线程开发中，如果子线程设定为了守护线程，守护线程会等待主线程运行完毕后被销毁。一个主线程可以设置多个守护线程，守护线程运行的前提是，主线程必须存在，如果主线程不存在了，守护线程会被销毁。

```python
 # 把子线程设置为守护线程，在启动线程前设置
thr.setDaemon(True)
thr.start()
```

###### 多线程的锁机制

多线程编程访问共享变量时会出现问题，但是多进程编程访问共享变量不会出现问题。因为多进程中，同一个变量各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享。 想实现多个线程共享变量，需要使用全局变量。在方法里加上全局关键字`global`定义全局变量，多线程才可以修改全局变量来共享变量。  多线程同时修改全局变量时会出现数据安全问题，线程不安全就是不提供数据访问保护，有可能出现多个线程先后更改数据造成所得到的数据是脏数据。 

在多线程情况下，所有的全局变量有所有线程共享。所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。在多线程情况下，使用全局变量并不会共享数据，会出现线程安全问题。线程安全就是多线程访问时，采用了加锁机制，当一个线程访问该类的某个数据时，进行保护，其他线程不能进行访问直到该线程读取完，其他线程才可使用。不会出现数据不一致针对线程安全问题，需要使用”互斥锁”，就像数据库里操纵数据一样，也需要使用锁机制。某个线程要更改共享数据时，先将其锁定，此时资源的状态为“锁定”，其他线程不能更改；直到该线程释放资源，将资源的状态变成“非锁定”，其他的线程才能再次锁定该资源。互斥锁保证了每次只有一个线程进行写入操作，从而保证了多线程情况下数据的正确性。

```python
mutex = threading.Lock()# 创建锁
mutex.acquire()# 锁定
mutex.release()# 释放
```

```python
def change(num, counter):
    global balance
    for i in range(counter):
        lock.acquire() # # 先要获取锁
        balance += num
        balance -= num
        lock.release()  # 释放锁
        if balance != 100:
            # 如果输出这句话，说明线程不安全
            print("balance=%d" % balance)
            break
```

 当某个线程执行change()函数时，通过`lock.acquire()`获取锁，那么其他线程就不能执行同步代码块了，只能等待知道锁被释放了，获得锁才能执行同步代码块。由于锁只有一个，无论多少线程，同一个时刻最多只有一个线程持有该锁，所以修改全局变量balance不会产生冲突。 

##### 垃圾回收

内存泄漏- 这里的泄漏，并不是说你的内存出现了信息安全问题，被恶意程序利用了，而是指程序本身没有设计好，导致程序未能释放已不再使用的内存。- 内存泄漏也不是指你的内存在物理上消失了，而是意味着代码在分配了某段内存后，因为设计错误，失去了对这段内存的控制，从而造成了内存的浪费。也就是这块内存脱离了gc的控制
因为python中一切皆为对象，你所看到的一切变量，本质上都是对象的一个指针。

当一个对象不再调用的时候，也就是当这个对象的引用计数（指针数）为 0 的时候，说明这个对象永不可达，自然它也就成为了垃圾，需要被回收。可以简单的理解为没有任何变量再指向它。
python针对循环引用，有它的自动垃圾回收算法1. 标记清除（mark-sweep）算法2. 分代收集（generational）
标记清除的步骤总结为如下步骤1. GC会把所有的『活动对象』打上标记2. 把那些没有标记的对象『非活动对象』进行回收
对于一个有向图，如果从一个节点出发进行遍历，并标记其经过的所有节点；那么，在遍历结束后，所有没有被标记的节点，我们就称之为不可达节点。显而易见，这些节点的存在是没有任何意义的，自然的，我们就需要对它们进行垃圾回收。

python同样给我们提供了手动释放内存的方法 gc.collect()
但是每次都遍历全图，对于 Python 而言是一种巨大的性能浪费。所以，在 Python 的垃圾回收实现中，mark-sweep 使用双向链表维护了一个数据结构，并且只考虑容器类的对象（只有容器类对象，list、dict、tuple，instance，才有可能产生循环引用）。
分代回收是一种以空间换时间的操作方式，Python将内存根据对象的存活时间划分为不同的集合，每个集合称为一个代，Python将内存分为了3“代”，分别为年轻代（第0代）、中年代（第1代）、老年代（第2代），他们对应的是3个链表，它们的垃圾收集频率与对象的存活时间的增大而减小。新创建的对象都会分配在年轻代，年轻代链表的总数达到上限时（当垃圾回收器中新增对象减去删除对象达到相应的阈值时），Python垃圾收集机制就会被触发，把那些可以被回收的对象回收掉，而那些不会回收的对象就会被移到中年代去，依此类推，老年代中的对象是存活时间最久的对象，甚至是存活于整个系统的生命周期内。同时，分代回收是建立在标记清除技术基础之上。

事实上，分代回收基于的思想是，新生的对象更有可能被垃圾回收，而存活更久的对象也有更高的概率继续存活。因此，通过这种做法，可以节约不少计算量，从而提高 Python 的性能。

所以对于刚刚的问题，引用计数只是触发gc的一个充分非必要条件，循环引用同样也会触发。





更新pip：在Linux或者OSX平台上执行命令 `pip install -U pip`、在Windows平台上执行命令`python -m pip install -U pip`

1. 从PyPI安装

	pip install SomePackage  #安装最新的版本
	pip install SomePackage==1.0.4  #安装指定版本
	pip install 'SomePackage>=1.0.4' #minimum版本

从PyPI用pip安装时，可能因为网络问题发生超时。此时可以指定国内镜像源，命令为：
`pip install SomePackage -i http://example.com`
常用的镜像源有: 清华：http://mirrors.tuna.tsinghua.edu.cn/pypi/simple、豆瓣：http://pypi.douban.com/simple 

2. 从Wheel安装: `pip install SomePackage-1.0-py2.py3-none-any.whl`

3. 从本地源代码安装通常源代码中都有`setup.py`文件：`python setup.py install`

查看某个已安装package

	pip show SomePackage #查看该package详细信息
	pip show --files SomePackage #查看该package已安装哪些文件

查看所有已经安装package

	pip list #列出所有已安装的package
	pip list --outdated #列出所有outdated package

3. 更新package: `pip install --upgrade SomePackage`

4. 卸载package: `pip uninstall SomePackage`

5. 寻找package：`pip search "query"`

6. pip freeze命令是输出当前已安装package的依赖列表，方便于复制当前已安装的package。可以通过重定向输出从而产生requirement文件，命令为： `pip freeze > requirements.txt`。你也可以从某个`requirements`文件中读取从而叠加到当前的`pip freeze`结果中：
    `pip freeze -r exist_requirements.txt > requirements.txt`。如果你是在`virtualenv`环境中，则通过 `pip freeze -l `命令则不会读取`globally-installed`包。如果你是使用`pip freeze --user`命令，则只会读取`user-site`目录中的包。如果有了`requirements`文件，则通过`pip install -r requirements.txt`命令可以安装所有的这些包

##### 三、配置文件

配置文件有三个级别：系统级别、用户级别、virtualenv级别。其读取顺序为：首先读取系统级别的配置文件，然后读取用户级别的配置文件，最后读取virtualenv级别的配置文件。如果同样的值在多个配置文件中设置，则最后读取的值会覆盖早期读取的值。

###### 2.用户级的的配置文件

用户级的pip配置文件位于：

* Unix系统中位于：`$HOME/.config/pip/pip.conf`（由`XDG_CONFIG_HOME`环境变量指定的），或者`$HOME/.pip/pip.conf`（优先级较低）中
* OSX系统中位于：`$HOME/Library/Application Support/pip/pip.conf`
* Windows系统中位于：`%APPDATA%\pip\pip.ini`或者`%HOME%\pip\pip.ini`（优先级较低）中

###### 3.virtualenv中的配置文件

你可以在env中设置不同虚拟环境下的配置文件：

* 在Unix/OSX系统中位于：`$VIRTUAL_ENV/pip.conf`
* 在Windows系统中位于： `%VIRTUAL_ENV%\pip.ini`

###### 4.配置文件的内容

		[global]   #针对所有pip命令的配置
		timeout = 60 #超时
		index-url = http://pypi.douban.com/simple #不同的库url
		trusted-host = pypi.douban.com        #添加豆瓣源为可信主机，要不然可能报错
		disable-pip-version-check = true      #取消pip版本检查，排除每次都报最新的pip
	
		[install] #针对具体的pip命令的配置
		ignore-installed = true
		no-dependencies = yes 



5.惯例： `scikit-learn estimator`遵守以下惯例：

- 除非显式指定数据类型，否则所有的输入数据都被转换成 `float64`
- 回归问题的输出被转换成 `float64`；分类问题的输出不被转换
- `estimator`的参数可以更新：
  - `estimator.set_params(...)`方法可以显式更新一个`estimator`的参数值
  - 多次调用`estimator.fit(...)`方法可以隐式更新一个`estimator`的参数值。最近的一次训练学到的参数会覆盖之前那次训练学到的参数值。

