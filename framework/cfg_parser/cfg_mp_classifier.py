from . import cfg_base

class MPClassifierDataLoaderConfigurator(cfg_base.BasicDataLoaderConfigurator):
    def __init__(self):
        super().__init__()

        self.noise_to_mask = float(0)

class ImageClassifierStructureConfigurator(cfg_base.BasicConvNetworkStructureConfigurator):
    def __init__(self):
        super().__init__()

        self.nc_f = int(32)
        self.nd_l = int(4)

        self.n_head = int(1)

class MPClassifierNetworkConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.network = ImageClassifierStructureConfigurator()    

class MPClassifierLossConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()

class MPClassifierSolverConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.param = str()

class MPClassifierRunnerConfigurator(cfg_base.BasicRunnerConfigurator):
    def __init__(self):
        super().__init__()

        self.data_loader_train = MPClassifierDataLoaderConfigurator()
        self.data_loader_val = MPClassifierDataLoaderConfigurator()
        self.data_loader_test = MPClassifierDataLoaderConfigurator()
        self.classifier = MPClassifierNetworkConfigurator()
        self.loss = MPClassifierLossConfigurator()
        self.solver = MPClassifierSolverConfigurator()