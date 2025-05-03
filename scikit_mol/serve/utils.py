from sklearn.pipeline import Pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.safeinference import SafeInferenceWrapper


def validate_pipeline(pipeline: Pipeline):
    first_step = pipeline.steps[0][1]
    predictor = pipeline.steps[-1][1]
    if not isinstance(first_step, (SmilesToMolTransformer)):
        raise ValueError(
            f"Pipeline must start with a SmilesToMolTransformer, not {type(first_step)}"
        )
    if not isinstance(predictor, SafeInferenceWrapper):
        raise ValueError(
            f"Pipeline must end with a SafeInferenceWrapper to enable safe inference, not {type(predictor)}"
        )
    return pipeline
