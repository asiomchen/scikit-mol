from typing import List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    smiles_list: List[str]


class PredictResponse(BaseModel):
    result: list[float]
    errors: dict[int, str]
