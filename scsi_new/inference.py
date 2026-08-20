import torch
from dataclasses import dataclass
import ode
import sde
from si import Interpolant, LinearInterpolant, GVPInterpolant

@dataclass
class Config_Inference:
    inference_style: str = "ode" # "ode" or "sde"
    noise_scheduler: str = "kl" # if `inference_style` is `sde`, then you can pick ["kl", "alpha"]
    t_min: float | None = None
    t_max: float | None = None

def build_inference(config_inference: Config_Inference, interpolant: Interpolant):
    '''
    Build function which takes in drift/score and outputs integrated scheme.
    '''
    inference_style = config_inference.config_inference
    if inference_style == 'ode':
        if interpolant is LinearInterpolant:
            t_min = default_when_none(0.0, t_min)
            t_max = default_when_none(1.0, t_max)
        elif interpolant is GVPInterpolant:
            t_min = default_when_none(0.0, t_min)
            t_max = default_when_none(1.0, t_max)
    elif inference_style == 'ode_Heun':
        if interpolant is LinearInterpolant:
            t_min = default_when_none(0.0, t_min)
            t_max = default_when_none(1.0, t_max)
        elif interpolant is GVPInterpolant:
            t_min = default_when_none(0.0, t_min)
            t_max = default_when_none(1.0, t_max)
    elif inference_style == 'sde_EulerMaruyama':
        if interpolant is LinearInterpolant:
            t_min = default_when_none(4e-2, t_min)
            t_max = default_when_none(1.0, t_max)
        elif interpolant is GVPInterpolant:
            t_min = default_when_none(4e-2, t_min)
            t_max = default_when_none(1.0, t_max)
    
    else:
        raise NotImplemented()

def default_when_none(value, x):
    if x is None:
        return value
    else:
        return x