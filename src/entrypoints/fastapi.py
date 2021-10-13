import time

from fastapi import FastAPI, Request, HTTPException, Depends
from sqlmodel import create_engine
from sqlalchemy.engine.base import Engine

from src.adapters.responses import MappingResponse, MappingResponseList
from src.service_layer import services
from src.service_layer.unit_of_work import SQLAlchemyUnitOfWork
from src.domain.tfidfmapping import StringNotFound
from src.config import settings


title = f"Accurity Automated Mappings API ({settings.DATABASE_TYPE})"
description = """
Accurity Automated Mappings API is a ML-based tool that generates mapping
candidates for either a single input string or set of strings, always using
the current/live database data for calculation.
"""

app = FastAPI(title=title, description=description, docs_url="/")


def get_engine(database: str) -> Engine:
    return create_engine(url=settings.ACCURITY_PROD_DB_SERVER_URI + "/" + database, echo=True)


@app.middleware("http")
async def add_response_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:,.2f} seconds"
    return response


@app.get('/data-fields/business-terms/one',
         response_model=MappingResponse,
         summary="Data fields vs Business terms mapping - ONE",
         tags=["Data fields vs. Business terms"])
def data_fields_business_terms_one(database: str, schema: str, data_field: str,
                                   engine=Depends(get_engine), ntop: int = 5):
    exec_dict = {"schema_translate_map": {"schema": schema}}
    try:
        model = services.map_data_fields_business_terms_one(
            data_field=data_field,
            ntop=ntop,
            uow=SQLAlchemyUnitOfWork(engine=engine, exec_dict=exec_dict)
        )
    except StringNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    return MappingResponse.from_model(input_string=data_field, model=model)


@app.get('/data-fields/business-terms/all',
         response_model=MappingResponseList,
         summary="Data fields vs Business terms mapping - ALL",
         tags=["Data fields vs. Business terms"])
def data_fields_business_terms_all(database: str, schema: str, engine=Depends(get_engine),
                                   ntop: int = 5):
    exec_dict = {"schema_translate_map": {"schema": schema}}
    model = services.map_data_fields_business_terms_all(
        ntop=ntop,
        uow=SQLAlchemyUnitOfWork(engine=engine, exec_dict=exec_dict)
    )
    return MappingResponseList.from_model(model=model)


@app.get('/data-structures/entities/one',
         response_model=MappingResponse,
         summary="Data structures vs Entities mapping - ONE",
         tags=["Data structures vs. Entities"])
def data_structures_entities_one(database: str, schema: str, data_structure: str,
                                 engine=Depends(get_engine), ntop: int = 5):
    exec_dict = {"schema_translate_map": {"schema": schema}}
    try:
        model = services.map_data_structures_entities_one(
            data_structure=data_structure,
            ntop=ntop,
            uow=SQLAlchemyUnitOfWork(engine=engine, exec_dict=exec_dict)
        )
    except StringNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    return MappingResponse.from_model(input_string=data_structure, model=model)


@app.get('/data-structures/entities/all',
         response_model=MappingResponseList,
         summary="Data structures vs Entities mapping - ALL",
         tags=["Data structures vs. Entities"])
def data_structures_entities_all(database: str, schema: str, engine=Depends(get_engine),
                                 ntop: int = 5):
    exec_dict = {"schema_translate_map": {"schema": schema}}
    model = services.map_data_structures_entities_all(
        ntop=ntop,
        uow=SQLAlchemyUnitOfWork(engine=engine, exec_dict=exec_dict)
    )
    return MappingResponseList.from_model(model=model)
