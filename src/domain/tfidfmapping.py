from typing import Set, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

from src.domain.basemapping import BaseMapping
from src.domain.stemmerdict import stemmer_dict as stemmer
from src.domain.stringcatalog import String, StringCatalog


class StringNotFound(Exception):
    pass


class TfidfDistanceMapping(BaseMapping):
    """Class for generating TFIDF-based string mappings.

    Parameters
    ----------
    ncandidates : int
        Number of candidates to generate for each input string
    input : str
        Input strings identifier (e.g. data_field)
    match : str
        Match strings identifier (e.g. business_term)
    model : str
        Model identifier

    Attributes
    ----------
    model_file : str
        Path where model will be saved
    output_data_dir : str
        Directory where output data can be stored
    """
    def __init__(self, ncandidates: int, input: str, match: str, model: str = 'MAP'):
        self.ncandidates = ncandidates
        self.input_catalog = StringCatalog(name=input)
        self.match_catalog = StringCatalog(name=match)
        self.model_file = '../models/tfidf_mapping_' + model.lower() + '.bin'
        self.output_data_dir = '../data/output/'

    def __repr__(self):
        return (
            f"TfidfDistanceMapping(model_file={self.model_file}," +
            f"output_data_dir={self.output_data_dir}, ncandidates={self.ncandidates})"
        )

    def load_data(self, input_strings: Set[str], match_strings: Set[str]):
        """Load data from strings and store them into input and match string catalogs."""
        self.input_catalog.add_strings(input_strings)
        self.match_catalog.add_strings(match_strings)

    def _check_analyzer(self, analyzer):
        if analyzer not in ['word', 'char', 'char_wb']:
            raise Exception("Analyzer must be one of the following: 'word', 'char', 'char_wb'.")

    def _print_vectorization_stats(self, csr_matrix, algo):
        print(f"TFIDF matrix statistics: {algo}")
        print(f"Dimensions: {csr_matrix.shape[0]:>3} rows, {csr_matrix.shape[1]} features/tokens")
        print(f"Sparsity: {csr_matrix.nnz / (csr_matrix.shape[0] * csr_matrix.shape[1]):>7,.2%}")
        print(f"# values: {csr_matrix.nnz:>7}")

    def _get_csr_topn_idx_data(self, csr_row, n):
        """Helper function to get list of (idx, data) tuples for top N matching candidates."""
        nnz = csr_row.getnnz()
        if nnz == 0:
            return None
        elif nnz <= n:
            result = zip(csr_row.indices, csr_row.data)
        else:
            idx = np.argpartition(csr_row.data, -n)[-n:]
            result = zip(csr_row.indices[idx], csr_row.data[idx])
        return sorted(result, key=lambda x: -x[1])

    def _scipy_cossim_topn(self, A, B, n):
        """Calculate top N matching candidates based on cosine similarity.

        Matching candidates are calculated for each entry in matrix A versus each entry in
        matrix B, using in both matrices the same number of pre-calculated TFIDF features.
        """
        A.astype(np.float32)
        B.astype(np.float32)
        C = round(awesome_cossim_topn(A, B, n, 0), 4)
        # C = round(A @ B, 4)
        return [self._get_csr_topn_idx_data(row, n) for row in C]

    def _adhoc_learned_data_stemmer(self, values: list, dictionary):
        keys = []
        for value in values:
            idx = None
            if value in dictionary:
                keys.append(value)
                continue
            for i, group in enumerate(list(dictionary.values())):
                if value in group:
                    idx = i
                    keys.append(list(dictionary.keys())[idx])
                    break
            if idx is None:
                keys.append(value)
        return keys

    def _stem_documents_adhoc(self, series, dictionary, sep=' '):
        out = []
        for row in series:
            input_list = row.split(sep)
            stemmed_list = self._adhoc_learned_data_stemmer(input_list, dictionary)
            out.append(sep.join(stemmed_list))
        return pd.Series(out)

    def find_mapping(self, ngrams=1, analyzer='word', use_stemmer=False, save=False,
                     input_string: str = None):
        """Calculate mapping candidates for each string from `input_catalog` w.r.t
        strings in `match_catalog` using L2-normalized TF-IDF vectorizer and cosine
        similarity metric.

        This method will calculate and store only `ncandidate` number of mapping
        candidates. Mapping candidates will be stored in `mapping_candidates[algo]`
        dictionary and optionally into xls file in the `output_data_dir` directory.

        Parameters
        ----------
        ngrams : int, optional
            Integer used for definition of n-grams, i.e. sequences of characters or word analysed
        """
        self._check_analyzer(analyzer)
        algo = 'Tfidf' + '-' + analyzer + '-' + str(ngrams) + '-GRAM-' + str(int(use_stemmer))

        # Strings used to learn the model are slugify-formatted pandas Series
        input_strings = self.input_catalog.to_df()['slugified']
        match_strings = self.match_catalog.to_df()['slugified']

        if use_stemmer:
            input_strings = self._stem_documents_adhoc(input_strings, stemmer)
            match_strings = self._stem_documents_adhoc(match_strings, stemmer)

        strings = pd.concat([input_strings, match_strings], axis=0)

        tfidf = TfidfVectorizer(input='content', lowercase=True, analyzer=analyzer,
                                stop_words=None, token_pattern=r'(?u)\b\w\w+\b',
                                ngram_range=(ngrams, ngrams), max_df=1.0, min_df=1, norm='l2')

        X = tfidf.fit_transform(strings)  # Learned document-term matrix
        self._print_vectorization_stats(X, algo)

        X_input_strings = X[:len(input_strings)]
        X_match_strings = X[len(input_strings):]

        cossim = self._scipy_cossim_topn(X_input_strings, X_match_strings.T, self.ncandidates)

        mapping_candidates: Dict[str, Tuple[str, float]] = {}
        input_catalog_dict = self.input_catalog.to_dict()
        match_catalog_dict = self.match_catalog.to_dict()

        for idx, candidates in enumerate(cossim):
            key = input_catalog_dict[idx]
            values = []
            if candidates:
                for candidate in candidates:
                    values.append((match_catalog_dict[candidate[0]], candidate[1]))
            mapping_candidates[key] = values

        if input_string and not mapping_candidates.get(input_string):
            raise StringNotFound(f"Input string {input_string} does not exist")

        self.mapping_candidates = mapping_candidates

        if save:
            self._save_mapping_to_xls(algo=algo, sheet_name=algo)
            self._save_model()
