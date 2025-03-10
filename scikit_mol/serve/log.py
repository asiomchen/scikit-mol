import copy
from typing import Protocol
from venv import logger

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import InvalidMol
from scikit_mol.standardizer import Standardizer


class LoggerProtocol(Protocol):
    def error():
        pass

    def info():
        pass

    def warning():
        pass

    def debug():
        pass

    def critical():
        pass


class InvalidMolsLoggingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, logger: LoggerProtocol):
        self.logger = logger
        self._last_info: dict[int, str] = None

    def transform(self, X, *args, **kwargs):
        self._last_info = {}
        tmp_X = X.flatten()
        for idx, mol in enumerate(tmp_X):
            if isinstance(mol, InvalidMol):
                msg = f"Invalid molecule at index {idx}. Error: {mol.error}"
                print(msg)
                logger.info(msg)
                self._last_info[idx] = mol.error
        return X


def add_pipeline_logging(
    pipeline: Pipeline, logging_transformer: InvalidMolsLoggingTransformer
):
    pipeline = copy.deepcopy(pipeline)
    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            f"Logging can be added only to Pipeline objects, not {type(pipeline)}"
        )
    insert_index = 0
    allowed_classes = (SmilesToMolTransformer, Standardizer)
    for step in pipeline.steps:
        if isinstance(step[1], allowed_classes):
            insert_index += 1
        else:
            break
    pipeline.steps.insert(insert_index, ("invalid_mols_logger", logging_transformer))
    return pipeline
