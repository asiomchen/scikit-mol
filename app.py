import logging
import pickle
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from scikit_mol.serve.log import InvalidMolsLoggingTransformer, add_pipeline_logging

app = FastAPI()

MODEL: Pipeline = pickle.load(open("pipeline.pkl", "rb"))  # noqa

logger = logging.getLogger("uvicorn.error")
log_transformer = InvalidMolsLoggingTransformer(logger)
MODEL = add_pipeline_logging(MODEL, log_transformer)


class PredictRequest(BaseModel):
    smiles_list: List[str]


class PredictResponse(BaseModel):
    result: list[float]
    errors: dict[int, str]


@app.get("/")
def read_root():
    return {"Hello": "World", "Model": str(MODEL)}


@app.post("/predict")
def predict(data: PredictRequest):
    result = list(MODEL.predict(data.smiles_list))
    result = [float(x) for x in result]
    print(log_transformer._last_info)
    return PredictResponse(result=result, errors=log_transformer._last_info)


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        i = 0
        while True:
            data = await websocket.receive_text()
            result = MODEL.predict([data])
            await websocket.send_text(str(result[0]))
            i += 1
    except WebSocketDisconnect:
        logger.info(f"Client disconnected after {i} messages")
        pass


def predict_proba(data: PredictRequest):
    result = list(MODEL.predict_proba(data.smiles_list)[0])
    result = [float(x) for x in result]
    return PredictResponse(result=result, errors=log_transformer._last_info)


if hasattr(MODEL, "predict_proba"):
    app.post("/predict_proba")(predict_proba)
