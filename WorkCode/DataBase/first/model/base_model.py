from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

engine = create_engine(
    'mysql+pymysql://root:011636@localhost:3306/pydata', echo=True)

