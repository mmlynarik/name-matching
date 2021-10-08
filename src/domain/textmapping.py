"""Text distance mapping class."""

import os
import logging
import pickle
import time
from datetime import datetime
from collections import namedtuple

import openpyxl as xls
import pandas as pd
from textdistance import jaro_winkler, jaro
from IPython.display import display

from src.utils.base import log_info, get_whitespaces, get_setvalued_dict
from src.domain.basemapping import BaseMapping
from src.csvreader import CSVReader, melt_mappings

Dataset = namedtuple("Dataset", ["business_terms", "data_fields", "validation_mappings"])


class TextDistanceMapping(BaseMapping, CSVReader):
    def __init__(self, name, raw_data_dir, sep=';', ncandidates=5, validation=False):
        self.name = name
        self.raw_data_dir = raw_data_dir
        self.model_file = '../models/text_mapping_' + name.lower() + '.bin'
        self.output_data_dir = '../data/output/'
        self.sep = sep
        self.ncandidates = ncandidates
        self.validation = validation
        self.mapping_candidates = {}
        self.feasible_algorithms = [
            # edit-based
            'Hamming', 'Levenshtein', 'DamerauLevenshtein', 'Jaro', 'JaroWinkler',
            'StrCmp95', 'NeedlemanWunsch', 'Gotoh', 'SmithWaterman',
            # token-based
            'Overlap', 'Cosine', 'Jaccard', 'Tversky', 'Sorensen'
        ]
        self.reverse_algorithms = [
            'Jaro', 'JaroWinkler', 'NeedlemanWunsch', 'SmithWaterman',
            'Gotoh', 'StrCmp95', 'Overlap', 'Jaccard', 'Cosine', 'Tversky', 'Sorensen'
        ]

    def __repr__(self):
        return (
            f"TextDistanceMappings(name={self.name}, validation={self.validation}, " +
            f"model_file={self.model_file}, raw_data_dir={self.raw_data_dir}, " +
            f"output_data_dir={self.output_data_dir}, ncandidates={self.ncandidates})"
        )

    def initialize_data(self):
        self.load_raw_data()
        self.calc_input_data()
        self.initialized = True

    def load_raw_data(self):
        """Load raw data from csv files and store them into pandas dataframes in data attribute.

        Use naming convention business_terms.csv for business terms and data_fields.csv for data
        fields. Already available data mappings can be used for validation. They must be
        stored in data_mappings.csv. Further conditions on naming convention of
        columns in csv files apply as well: business_terms.csv must contain columns
        `entity` and `attribute`, data_fields.csv must contain columns
        `data_field` and `data_structure` and data_mappings.csv must contain columns
        `entity`, `attribute` and `mapping`.
        """
        data = []
        for name, mandatory, func in zip(["business_terms", "data_fields", "validation_mappings"],
                                         [True, True, True if self.validation else False],
                                         [None, None, melt_mappings]):
            csv = CSVReader(name=name, file_dir=self.raw_data_dir,
                            mandatory=mandatory, sep=self.sep)
            df = csv.load_transform(func)
            data.append(df)

        self.raw = Dataset(*data)

    def calc_input_data(self):
        business_terms = (
            self.raw.business_terms.entity.str.replace(' ', '_') + '.' +
            self.raw.business_terms.attribute.str.replace(' ', '_')
        ).str.lower().drop_duplicates().sort_values().reset_index(drop=True)

        if self.validation:
            data_fields = self.raw.validation_mappings.data_field.drop_duplicates()
        else:
            data_fields = (
                self.raw.data_fields.data_structure.str.replace(' ', '_') + '.' +
                self.raw.data_fields.data_field.str.replace(' ', '_')
            ).str.lower().drop_duplicates().sort_values().reset_index(drop=True)

        validation_mappings = None
        if self.raw.validation_mappings is not None:
            validation_mappings = get_setvalued_dict(self.raw.validation_mappings.data_field,
                                                     self.raw.validation_mappings.business_term)

        self.input = Dataset(business_terms, data_fields, validation_mappings)

    def _check_algorithm(self, algo):
        if algo not in self.feasible_algorithms:
            raise Exception(f"{algo} is not among feasible distance algorithms.")

    def find_mapping(self, algorithm, ngrams=1, save=True):
        """Calculate mapping candidates for each data field w.r.t the business terms
           using selected algorithm.

        This method will calculate distances between data fields and business terms and
        for each data field will store only `ncandidate` number of mapping candidates.
        The mapping candidates will be stored in `self.mapping_candidates[algo]` dict
        and into xls file in the `self.output_data_dir` directory.

        Parameters
        ----------
        algorithm : algorithm instance from textdistance library
            Callable class instance from textdistance library  calculating distance between
            data fields and business terms.
        ngrams : int, optional
            Integer used for definition of n-grams, i.e. sequences of characters analysed
        """
        algo_short = algorithm.__class__.__name__
        algo = algo_short + '-' + str(ngrams) + '-GRAM-0'
        self._check_algorithm(algo_short)
        self._check_initialized()
        sign = -1 if algo_short in self.reverse_algorithms else 1
        algorithm.qval = ngrams

        mapping_candidates = {}

        start_time = datetime.now()
        logging.basicConfig(level=logging.DEBUG)
        log_info(self.name,
                 f""" Start calculating mappings using {algo}, with sign={sign}: """ +
                 f"""{ start_time.strftime('%X')}""")

        for data_field in self.input.data_fields:
            distances = []
            for business_term in self.input.business_terms:
                distance = round(algorithm(data_field, business_term), 4)
                distances.append((business_term, distance))
            distances = sorted(distances, key=lambda x: sign * x[1])
            mapping_candidates[data_field] = distances[:self.ncandidates]

        self.mapping_candidates[algo] = mapping_candidates
        sheet_name = algo
        if save:
            self._save_mapping_to_xls(algo, sheet_name)
            self._save_model()
        log_info(self.name,
                 f""" Finished mappings for data_fields in {datetime.now() - start_time}""")
