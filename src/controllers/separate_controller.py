from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th

#multi-agent controller with separete parameters for each agent.
class SeparateMAC(BasicMAC):
    def __init__(self,scheme, groups, args):
        super(SeparateMAC,self).__init__(scheme, groups, args)


#TODO: complete it.