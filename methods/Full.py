from .Uniform import Uniform

class Full(Uniform):
    method_name = 'Full'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.ratio = 1.0