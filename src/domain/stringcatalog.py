from dataclasses import dataclass
from typing import Set

import pandas as pd
import slugify as slg


@dataclass(frozen=True)
class String:
    """Dataclass to represent string, its index and transformed variant."""
    idx: str
    raw: str

    @property
    def slugified(self):
        return slg.slugify(self.raw, separator=" ")


class StringCatalog:
    """Data structure for storing and handling set of String objects."""
    def __init__(self, name: str) -> None:
        self._strings: set[String] = set()
        self.name = name

    def __len__(self) -> int:
        return len(self._strings)

    def __repr__(self) -> str:
        return f"{self.name} with {len(self)} strings"

    def to_dict(self) -> dict:
        """Convert catalog to dict with key/value pair equal to String's idx/raw."""
        return {s.idx: s.raw for s in self._strings}

    def to_df(self) -> pd.DataFrame:
        """Convert catalog to dataframe with index equal to String's idx."""
        df = pd.DataFrame(data=[(s.idx, s.raw, s.slugified) for s in self._strings],
                          columns=['idx', 'raw', 'slugified'])
        return df.set_index('idx').sort_index()

    def get_string(self, idx: int) -> str:
        """Get raw string from catalog by index."""
        for string in self._strings:
            if string.idx == idx:
                return string.raw

    def add_strings(self, strings: Set[str]) -> None:
        """Add strings to catalog from a set."""
        strings_to_add = strings - set(self.to_dict().values())
        for string in strings_to_add:
            self._strings.add(String(raw=string, idx=len(self)))
