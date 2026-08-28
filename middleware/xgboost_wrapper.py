"""
Wrapper de XGBoost como torch.nn.Module.

CONTEXTO: la librería oficial FastSHAP (iancovert/fastshap) exige que el
"modelo a explicar" pasado a MarginalImputer/BaselineImputer sea un
torch.nn.Module real (internamente llama a next(model.parameters()) para
determinar el device). El modelo subrogado de la Fase 2 del documento de
tesis es XGBoost, que NO es un módulo de PyTorch.

Esta clase envuelve un modelo XGBoost ya entrenado en una interfaz
compatible con torch.nn.Module, delegando la predicción real al XGBoost
por debajo. No introduce ninguna red neuronal "surrogate" adicional para
imputación (a diferencia del notebook de ejemplo census.ipynb del
repositorio oficial, que usa una red separada para datos donde el modelo
original también es una red); aquí el modelo a explicar sigue siendo
exactamente el XGBoost que exige el documento de tesis, no una
aproximación neuronal de él.

Tiene un único parámetro "dummy" registrado SOLO para que
next(model.parameters()) no falle al ser llamado por la librería externa;
ese parámetro no participa en ningún cálculo ni se entrena.
"""

import numpy as np
import torch
import torch.nn as nn


class XGBoostWrapper(nn.Module):
    """
    Envuelve un modelo XGBoost (xgboost.XGBRegressor ya entrenado) para que
    pueda usarse como "modelo a explicar" dentro de MarginalImputer.
    """

    def __init__(self, modelo_xgboost):
        super().__init__()
        self.modelo_xgboost = modelo_xgboost
        # Parámetro dummy: necesario únicamente para que
        # next(self.parameters()) tenga algo que devolver (lo exige
        # internamente MarginalImputer/BaselineImputer del repositorio
        # oficial para determinar el device). No se usa en forward().
        self._parametro_dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor (batch, n_dimensiones) con variables de decisión.
        Retorna: tensor (batch, 1) con el fitness predicho por XGBoost.

        La predicción de XGBoost ocurre en CPU/numpy (XGBoost no opera
        sobre tensores de PyTorch), por lo que se hace la conversión
        ida y vuelta explícitamente. Esto no participa en ningún grafo
        de autograd —correcto, porque el XGBoost no se entrena aquí,
        solo se evalúa como función fija dentro del imputer.
        """
        x_numpy = x.detach().cpu().numpy()
        pred_numpy = self.modelo_xgboost.predict(x_numpy)
        pred_tensor = torch.tensor(
            pred_numpy, dtype=torch.float32, device=x.device
        ).reshape(-1, 1)
        return pred_tensor