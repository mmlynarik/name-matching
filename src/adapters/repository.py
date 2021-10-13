# Repository pattern

from abc import ABC, abstractmethod
from typing import Set, Dict

from sqlmodel import select, Session

from src.config import settings

if settings.DATABASE_TYPE == "POSTGRES":
    from src.adapters import orm
else:
    from src.adapters import legacy_orm as orm


class Repository(ABC):

    @abstractmethod
    def get_data_fields(self):
        return

    @abstractmethod
    def get_data_structures(self):
        return

    @abstractmethod
    def get_entities(self):
        return

    @abstractmethod
    def get_business_terms(self):
        return


class SQLAlchemyRepository(Repository):

    def __init__(self, session: Session, exec_dict: Dict):
        self.session = session
        self.exec_dict = exec_dict

    def get_data_fields(self) -> Set[str]:
        stmt = select(orm.DataStructure, orm.DataField).join(orm.DataField)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name + "." + df.name for ds, df in results}

    def get_business_terms(self) -> Set[str]:
        stmt = (
            select(orm.Attribute, orm.Entity, orm.AttributeDefinition)
            .join(orm.Entity)
            .join(orm.AttributeDefinition)
        )
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name + "." + ad.name for _, e, ad in results}

    def get_data_structures(self) -> Set[str]:
        stmt = select(orm.DataStructure)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name for ds in results}

    def get_entities(self) -> Set[str]:
        stmt = select(orm.Entity)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name for e in results}
