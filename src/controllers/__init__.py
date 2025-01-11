REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .dmcg_controller import DeepCoordinationGraphMAC
REGISTRY["dmcg_mac"] = DeepCoordinationGraphMAC

