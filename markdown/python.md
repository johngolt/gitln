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

