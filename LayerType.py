from enum import Enum


class LayerType(Enum):

    SEQUENTIAL = 0
    SKIP = 1

    @staticmethod
    def layer_name_to_layer_type(layer_name):
        if layer_name.split(".")[1] == "1":
            return LayerType.SEQUENTIAL
        else:
            return LayerType.SKIP
