# Legacy SQLModel ORM

from sqlmodel import Field, SQLModel


class DataField(SQLModel, table=True):
    __tablename__ = "data_field"
    __table_args__ = {"schema": "schema"}

    uuid: str = Field(..., primary_key=True)
    name: str
    data_structure_uuid: str = Field(..., foreign_key="schema.data_structure.uuid")


class DataStructure(SQLModel, table=True):
    __tablename__ = "data_structure"
    __table_args__ = {"schema": "schema"}

    uuid: str = Field(..., primary_key=True)
    name: str


class Entity(SQLModel, table=True):
    __tablename__ = "entity"
    __table_args__ = {"schema": "schema"}

    uuid: str = Field(..., primary_key=True)
    name: str


class Attribute(SQLModel, table=True):
    __tablename__ = "attribute"
    __table_args__ = {"schema": "schema"}

    uuid: str = Field(..., primary_key=True)
    name: str
    attribute_definition_uuid: str = Field(..., foreign_key="schema.attribute_definition.uuid")
    entity_uuid: str = Field(..., foreign_key="schema.entity.uuid")


class AttributeDefinition(SQLModel, table=True):
    __tablename__ = "attribute_definition"
    __table_args__ = {"schema": "schema"}

    uuid: str = Field(..., primary_key=True)
    name: str
