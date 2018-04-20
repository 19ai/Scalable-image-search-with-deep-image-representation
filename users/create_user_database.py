import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import csv

engine = create_engine('sqlite:///painting.db', echo=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)

    def __init__(self, username, password):
        """"""
        self.username = username
        self.password = password

def create_database(database_file):
    Base.metadata.create_all(engine)

    # create a Session
    Session = sessionmaker(bind=engine)
    session = Session()

    with open(database_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        reader.next()
        for line in reader:
            user = User(line[0], line[1])
            session.add(user)

    session.commit()

if __name__ == "__main__":
    create_database(database_file = 'user_database,csv')