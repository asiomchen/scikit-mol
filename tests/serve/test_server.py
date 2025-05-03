from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.safeinference import SafeInferenceWrapper
from scikit_mol.serve.models import PredictRequest
from scikit_mol.serve.server import ScikitMolServer


@pytest.fixture
def mock_pipeline():
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer()),
            ("fingerprint", MorganFingerprintTransformer(radius=2, fpSize=256)),
            (
                "model",
                SafeInferenceWrapper(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    replace_value=-1000,
                ),
            ),
        ]
    )
    pipeline.fit(["CCO", "CCN", "CCC", "CCCC", "CCOC"], [1, 0, 0, 0, 1])
    pipeline.predict = MagicMock()
    pipeline.predict_proba = MagicMock()
    pipeline.predict.return_value = [0.0]
    pipeline.predict_proba.return_value = [[0.3, 0.7]]
    return pipeline


@pytest.fixture
def server(mock_pipeline):
    return ScikitMolServer(mock_pipeline)


@pytest.fixture
def test_app(server):
    app = server._create_app()
    return app


def test_init_with_invalid_model():
    with pytest.raises(FileNotFoundError):
        ScikitMolServer("not_a_pipeline")


def test_init_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        ScikitMolServer("nonexistent.pkl")


def test_create_app(test_app):
    client = TestClient(test_app)
    response = client.get("/")
    assert response.status_code == 200


def test_predict(test_app):
    client = TestClient(test_app)
    request = PredictRequest(smiles_list=["CC"])
    response = client.post("/predict", json=request.model_dump())
    assert response.status_code == 200
    assert "result" in response.json()


def test_predict_proba(test_app):
    client = TestClient(test_app)
    request = PredictRequest(smiles_list=["CC"])
    response = client.post("/predict_proba", json=request.model_dump())
    assert response.status_code == 200
    assert "result" in response.json()


@pytest.mark.anyio
async def test_ws_predict(test_app):
    client = TestClient(test_app)
    with client.websocket_connect("/ws/predict") as websocket:
        websocket.send_text("CC")
        data = websocket.receive_text()
        assert data == "0.0"


@pytest.mark.anyio
async def test_ws_predict_proba(test_app):
    client = TestClient(test_app)
    with client.websocket_connect("/ws/predict_proba") as websocket:
        websocket.send_text("CC")
        data = websocket.receive_text()
        assert data == "[0.3, 0.7]"


def test_read_root(test_app):
    client = TestClient(test_app)
    response = client.get("/")
    data = response.json()
    assert response.status_code == 200
    assert "Model" in data
    assert "ScikitMol version" in data
    assert "Python version" in data
