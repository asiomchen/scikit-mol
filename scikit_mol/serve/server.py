import logging
import os
import pickle
from typing import Union

import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from sklearn.pipeline import Pipeline

from scikit_mol._version import __version__ as scikit_mol_version
from scikit_mol.safeinference import set_safe_inference_mode
from scikit_mol.serve.log import InvalidMolsLoggingTransformer, add_pipeline_logging
from .utils import validate_pipeline
from .models import PredictRequest, PredictResponse

class ScikitMolServer:
    def __init__(self, model: Union[Pipeline, str]):
        if isinstance(model, str):
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model file {model} does not exist.")
            model = pickle.load(open(model, "rb"))
        if not isinstance(model, Pipeline):
            raise ValueError(f"Model must be a Pipeline, not {type(model)}")
        self.model = validate_pipeline(model)
        self.logger = logging.getLogger("uvicorn.error")
        self.log_transformer: InvalidMolsLoggingTransformer = None
        if isinstance(model, str):
            self.model = pickle.load(open(model, "rb"))

    def _create_app(self) -> FastAPI:
        app = FastAPI()
        router = APIRouter()
        app.add_api_route("/", self._read_root, methods=["GET"])
        router.add_api_route("/predict", self._predict, methods=["POST"], response_model=PredictResponse)
        router.add_api_websocket_route("/ws/predict", self._ws_predict)
        router.add_api_route("/predict_proba", self._predict_proba, methods=["POST"], response_model=PredictResponse)
        router.add_api_websocket_route("/ws/predict_proba", self._ws_predict_proba)
        app.include_router(router)
        return app
    
    def run(self, host: str = "localhost", port: int = 8000):
        app = self._create_app()
        logger = logging.getLogger("uvicorn.error")
        self.log_transformer = InvalidMolsLoggingTransformer(logger)
        self.model = add_pipeline_logging(self.model, self.log_transformer)
        set_safe_inference_mode(self.model, True)
        uvicorn.run(app, host=host, port=port)
        return app
    
    def _read_root(self):
        return {"Model": str(self.model), 
                "ScikitMol version": scikit_mol_version, 
                "Python version": os.sys.version}

    def _predict(self, data: PredictRequest):
        result = list(self.model.predict(data.smiles_list))
        result = [float(x) for x in result]
        self.logger.error(self.log_transformer._last_info)
        return PredictResponse(result=result, errors=self.log_transformer._last_info)

    async def _ws_predict(self, websocket: WebSocket):
        await websocket.accept()
        try:
            i = 0
            while True:
                data = await websocket.receive_text()
                result = self.model.predict([data])
                await websocket.send_text(str(result[0]))
                i += 1
        except WebSocketDisconnect:
            self.logger.info(f"Client disconnected after {i} messages")
            pass

    def _predict_proba(self, data: PredictRequest):
        result = list(self.model.predict_proba(data.smiles_list)[0])
        result = [float(x) for x in result]
        return PredictResponse(result=result, errors=self.log_transformer._last_info)
    
    async def _ws_predict_proba(self, websocket: WebSocket):
        await websocket.accept()
        try:
            i = 0
            while True:
                data = await websocket.receive_text()
                result = self.model.predict_proba([data])
                await websocket.send_text(str(result[0]))
                i += 1
        except WebSocketDisconnect:
            self.logger.info(f"Client disconnected after {i} messages")
            pass
