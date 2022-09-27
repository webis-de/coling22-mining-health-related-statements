import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Article:
    pm_id: int = 0
    title: str = ""
    abstract: str = ""
    journal: str = ""
    authors: Optional[List[str]] = None
    language: str = ""
    publication_types: Optional[List[str]] = None
    mesh_headings: Optional[List[str]] = None
    medline_date: Optional[datetime.date] = None

    def update(self, key: str, value: Any) -> None:
        if not value:
            return
        if key == "year":
            self.medline_date = datetime.date(1900, 1, 1)
            self.medline_date = datetime.date(
                int(value), self.medline_date.month, self.medline_date.day
            )
            return
        elif key == "month":
            if self.medline_date is None:
                self.medline_date = datetime.date(1900, 1, 1)
            self.medline_date = datetime.date(
                self.medline_date.year, int(value), self.medline_date.day
            )
            return
        elif key == "day":
            if self.medline_date is None:
                self.medline_date = datetime.date(1900, 1, 1)
            self.medline_date = datetime.date(
                self.medline_date.year, self.medline_date.month, int(value)
            )
            return

        variable = key
        if variable in ("first_name", "last_name"):
            variable = "authors"
        prev_value = self.__dict__[variable]
        if prev_value is None:
            self.__dict__[variable] = prev_value = []

        if key == "first_name":
            if not self.authors:
                return
            value = prev_value[-1] + f", {value}"
            del prev_value[-1]
        elif key == "qualifiers":
            assert self.mesh_headings
            self.mesh_headings[-1] += f" {value}"
        if isinstance(prev_value, list):
            self.__dict__[variable].append(value)
        else:
            if isinstance(prev_value, str) and prev_value:
                self.__dict__[variable] += " "
            self.__dict__[variable] += type(prev_value)(value)

    def to_json(self) -> Dict[str, Any]:
        return {
            "pm_id": self.pm_id,
            "title": self.title,
            "abstract": self.abstract,
            "journal": self.journal,
            "language": self.language,
            "publication_types": self.publication_types,
            "mesh_headings": self.mesh_headings,
            "medline_date": str(self.medline_date),
            "authors": self.authors,
        }
