# Unit of Work pattern

from abc import ABC, abstractmethod

from sqlmodel import Session

from src.adapters.repository import Repository, SQLAlchemyRepository


class UnitOfWork(ABC):
    repo: Repository

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, *args):
        return

    @abstractmethod
    def commit(self):
        return


class SQLAlchemyUnitOfWork(UnitOfWork):
    def __init__(self, engine, exec_dict):
        self.engine = engine
        self.exec_dict = exec_dict

    def __enter__(self):
        self.session = Session(self.engine)
        self.repo = SQLAlchemyRepository(self.session, self.exec_dict)
        return super().__enter__()

    def __exit__(self, *args):
        self.session.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()
