# Repository pattern

from abc import ABC, abstractmethod
from typing import Set, Dict

from sqlmodel import select, Session


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


class LegacyRepository(Repository):

    def __init__(self, session: Session, exec_dict: Dict):
        self.session = session
        self.exec_dict = exec_dict

    def get_data_fields(self) -> Set[str]:
        from src.adapters.legacy_orm import DataStructure, DataField
        stmt = select(DataStructure, DataField).join(DataField)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name + "." + df.name for ds, df in results}

    def get_business_terms(self) -> Set[str]:
        from src.adapters.legacy_orm import Entity, Attribute, AttributeDefinition
        stmt = select(Attribute, Entity, AttributeDefinition).join(Entity).join(AttributeDefinition)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name + "." + ad.name for _, e, ad in results}

    def get_data_structures(self) -> Set[str]:
        from src.adapters.legacy_orm import DataStructure
        stmt = select(DataStructure)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name for ds in results}

    def get_entities(self) -> Set[str]:
        from src.adapters.legacy_orm import Entity
        stmt = select(Entity)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name for e in results}


class PostgresRepository(Repository):

    def __init__(self, session: Session, exec_dict: Dict):
        self.session = session
        self.exec_dict = exec_dict

    def get_data_fields(self) -> Set[str]:
        from src.adapters.orm import DataStructure, DataField
        stmt = select(DataStructure, DataField).join(DataField)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name + "." + df.name for ds, df in results}

    def get_business_terms(self) -> Set[str]:
        from src.adapters.orm import Entity, Attribute, AttributeDefinition
        stmt = select(Attribute, Entity, AttributeDefinition).join(Entity).join(AttributeDefinition)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name + "." + ad.name for _, e, ad in results}

    def get_data_structures(self) -> Set[str]:
        from src.adapters.orm import DataStructure
        stmt = select(DataStructure)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {ds.name for ds in results}

    def get_entities(self) -> Set[str]:
        from src.adapters.orm import Entity
        stmt = select(Entity)
        results = self.session.exec(stmt, execution_options=self.exec_dict)
        return {e.name for e in results}
