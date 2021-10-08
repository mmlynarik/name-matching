# Pydantic data models

from typing import List

from pydantic import BaseModel

from src.domain.tfidfmapping import TfidfDistanceMapping


class HTTPError(BaseModel):
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "Input string does not exist"},
        }


class MappingResult(BaseModel):
    input_string: str
    match_strings: List[str]
    match_similarities: List[float]


class MappingResponse(BaseModel):
    input_type: str
    match_type: str
    mappings: MappingResult

    @classmethod
    def from_model(cls, input_string: str, model: TfidfDistanceMapping):
        mapping_result = MappingResult(
            input_string=input_string,
            match_strings=[x[0] for x in model.mapping_candidates[input_string]],
            match_similarities=[x[1] for x in model.mapping_candidates[input_string]]
        )
        return {
            "input_type": model.input_catalog.name,
            "match_type": model.match_catalog.name,
            "mappings": mapping_result
        }


class MappingResponseList(BaseModel):
    input_type: str
    match_type: str
    mappings: List[MappingResult]

    @classmethod
    def from_model(cls, model: TfidfDistanceMapping):
        mapping_results = []
        for s in list(model.mapping_candidates.keys()):
            mapping_results.append(
                MappingResult(
                    input_string=s,
                    match_strings=[x[0] for x in model.mapping_candidates[s]],
                    match_similarities=[x[1] for x in model.mapping_candidates[s]]
                )
            )
        return {
            "input_type": model.input_catalog.name,
            "match_type": model.match_catalog.name,
            "mappings": mapping_results
        }
