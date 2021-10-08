# Unit of Work pattern

from abc import ABC, abstractmethod

from sqlmodel import Session

from src.adapters.repository import Repository


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


class PostgresUnitOfWork(UnitOfWork):
    def __init__(self, engine, exec_dict):
        self.engine = engine
        self.exec_dict = exec_dict

    def __enter__(self):
        from src.adapters.repository import PostgresRepository
        self.session = Session(self.engine)
        self.repo = PostgresRepository(self.session, self.exec_dict)
        return super().__enter__()

    def __exit__(self, *args):
        self.session.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()


class LegacyUnitOfWork(UnitOfWork):
    def __init__(self, engine, exec_dict):
        self.engine = engine
        self.exec_dict = exec_dict

    def __enter__(self):
        from src.adapters.repository import LegacyRepository
        self.session = Session(self.engine)
        self.repo = LegacyRepository(self.session, self.exec_dict)
        return super().__enter__()

    def __exit__(self, *args):
        self.session.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()
