"""
"""

class RRTrainerBase(object):
    def __init__(self, cfg, env, model, prepr, rmem, *args, **kwargs):
        self.cfg = cfg
        self.env = env
        self.model = model

        self.prepr = prepr
        self.rmem = rmem
