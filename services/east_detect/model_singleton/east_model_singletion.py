import threading

import copy
import keras
from services.east_detect.model_singleton.east_model import EAST_model

class EASTModel:
  __east_model = None
  
  @staticmethod
  def getModel():
    if EASTModel.__east_model is None:
      with threading.Lock():
        if EASTModel.__east_model is None:
          EASTModel.__east_model = EAST_model()

    east_model = copy.deepcopy(EASTModel.__east_model)
    east_model.model = keras.models.clone_model(EASTModel.__east_model.model)

    return east_model
