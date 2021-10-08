from typing import Tuple, Dict

import pytest
from fastapi.testclient import TestClient
from sqlmodel import create_engine

from src.entrypoints.fastapi import app, get_engine
from src.config import settings


# Use oracle server to align with input_params_fixture below
async def override_get_engine_dependency(database: str):
    return create_engine(url=settings.ACCURITY_TEST_DB_SERVER_URI + "/" + database, echo=True)


client = TestClient(app)


app.dependency_overrides[get_engine] = override_get_engine_dependency


@pytest.fixture(name="inputs")
def input_strings_fixture():
    data_field = "ACCOUNTING_METHOD_DIM.ACCOUNTING_METHOD_SK"
    data_structure = "ACCOUNTING_METHOD_DIM"
    return data_field, data_structure


@pytest.fixture(name="params")
def input_params_fixture():
    return {
        "database": "orclcdb",
        "schema": "glosar",
        "ntop": 5
    }


def test_data_fields_business_terms_one(inputs: Tuple[str], params: Dict):
    params_dict = {**params, "data_field": inputs[0]}
    r = client.get("/data-fields/business-terms/one", params=params_dict)
    assert r.status_code == 200
    r = r.json()
    assert r["input_type"] == "data_field"
    assert r["match_type"] == "business_term"
    assert r["mappings"]["input_string"] == inputs[0]
    assert len(r["mappings"]["match_strings"]) == 5
    assert len(r["mappings"]["match_similarities"]) == 5


def test_data_fields_business_terms_all(params: Dict):
    r = client.get("/data-fields/business-terms/all", params=params)
    assert r.status_code == 200
    r = r.json()
    assert r["input_type"] == "data_field"
    assert r["match_type"] == "business_term"
    assert len(r["mappings"][0]["match_strings"]) == 5
    assert len(r["mappings"][0]["match_similarities"]) == 5


def test_data_structures_entities_one(inputs: Tuple[str], params: Dict):
    params_dict = {**params, "data_structure": inputs[1]}
    r = client.get("/data-structures/entities/one", params=params_dict)
    assert r.status_code == 200
    r = r.json()
    assert r["input_type"] == "data_structure"
    assert r["match_type"] == "entity"
    assert r["mappings"]["input_string"] == inputs[1]
    assert len(r["mappings"]["match_strings"]) == 5
    assert len(r["mappings"]["match_similarities"]) == 5


def test_data_structures_entities_all(params: Dict):
    r = client.get("/data-structures/entities/all", params=params)
    assert r.status_code == 200
    r = r.json()
    assert r["input_type"] == "data_structure"
    assert r["match_type"] == "entity"
    assert len(r["mappings"][0]["match_strings"]) == 5
    assert len(r["mappings"][0]["match_similarities"]) == 5
