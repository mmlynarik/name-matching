# Service layer functions

from src.domain.tfidfmapping import TfidfDistanceMapping
from src.service_layer.unit_of_work import UnitOfWork


def map_data_fields_business_terms_one(data_field: str, ntop: int, uow: UnitOfWork):
    with uow:
        input_strings = uow.repo.get_data_fields()
        match_strings = uow.repo.get_business_terms()

    model = TfidfDistanceMapping(ncandidates=ntop, input="data_field", match="business_term")
    model.load_data(input_strings, match_strings)
    model.find_mapping(ngrams=3, analyzer='char_wb', use_stemmer=True, input_string=data_field)
    return model


def map_data_fields_business_terms_all(ntop: int, uow: UnitOfWork):
    with uow:
        input_strings = uow.repo.get_data_fields()
        match_strings = uow.repo.get_business_terms()

    model = TfidfDistanceMapping(ncandidates=ntop, input="data_field", match="business_term")
    model.load_data(input_strings, match_strings)
    model.find_mapping(ngrams=3, analyzer='char_wb', use_stemmer=True)
    return model


def map_data_structures_entities_one(data_structure: str, ntop: int, uow: UnitOfWork):
    with uow:
        input_strings = uow.repo.get_data_structures()
        match_strings = uow.repo.get_entities()

    model = TfidfDistanceMapping(ncandidates=ntop, input="data_structure", match="entity")
    model.load_data(input_strings, match_strings)
    model.find_mapping(ngrams=3, analyzer='char_wb', use_stemmer=True, input_string=data_structure)
    return model


def map_data_structures_entities_all(ntop: int, uow: UnitOfWork):
    with uow:
        input_strings = uow.repo.get_data_structures()
        match_strings = uow.repo.get_entities()

    model = TfidfDistanceMapping(ncandidates=ntop, input="data_structure", match="entity")
    model.load_data(input_strings, match_strings)
    model.find_mapping(ngrams=3, analyzer='char_wb', use_stemmer=True)
    return model
