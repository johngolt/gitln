'''
ORM即Object Relational Mapper，可以简单理解为数据库表和Python类之间的映射，通过操作Python类，
可以间接操作数据库。
引进ORM框架时，我的项目会参考MVC模式做以下设计。其中model存储的是一些数据库模型，
即数据库表映射的Python类；model_op存储的是每个模型对应的操作，
即增删查改；调用方（如main.py）执行数据库操作时，只需要调用model_op层，
并不用关心model层，从而实现解耦。
'''

from model.base_model import Base, engine

if __name__ == '__main__':
    Base.metadata.create_all(engine)

