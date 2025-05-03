from scikit_mol.standardizer import Standardizer
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scikit_mol.safeinference import SafeInferenceWrapper
import pickle

pipeline = Pipeline([
    ("smiles_to_mol", SmilesToMolTransformer()),
    ("standardizer", Standardizer()),
    ("fingerprint", MorganFingerprintTransformer(radius=2, nBits=256)),
    ("model", SafeInferenceWrapper(RandomForestRegressor(n_estimators=100, random_state=42), replace_value=-1000)),
])
pipeline.fit(
    ["CCO", "CCN", "CCC", "CCCC"],
    [2,2,3,4]
)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Pipeline saved to pipeline.pkl")
