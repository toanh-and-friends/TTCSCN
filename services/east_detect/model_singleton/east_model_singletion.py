import threading

import keras
from services.east_detect.model_singleton.east_model import EAST_model

class EASTModel:
    __model = None
  
    @staticmethod
    def getModel():
        if EASTModel.__model is None:
            with threading.Lock():
                if EASTModel.__model is None:
                    EASTModel.__model = EAST_model()

        model = keras.models.clone_model(EASTModel.__model)
      
        return model
