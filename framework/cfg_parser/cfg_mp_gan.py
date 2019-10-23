from . import cfg_base

class MPGANDataLoaderConfigurator(cfg_base.BasicDataLoaderConfigurator):
    def __init__(self):
        super().__init__()

        self.noise_to_mask = float(0.05)
        #self.voxel_order = str('NDHWC')

class VoxelGeneratorStructureConfigurator(cfg_base.BasicConvNetworkStructureConfigurator):
    def __init__(self):
        super().__init__()

        self.sn = int(0)

        self.z_dim = int(256)
        self.nc_f = int(512)
        self.nd_f = int(4)
        self.out_dim = int(32)

        self.voxel_order = str('NDHWC')

        self.mirror = int(1)

class ImageDiscriminatorStructureConfigurator(cfg_base.BasicConvNetworkStructureConfigurator):
    def __init__(self):
        super().__init__()

        self.sn = int(0)

        self.nc_f = int(32)
        self.nd_l = int(4)

        self.n_head = int(1)
        self.n_shared = int(2)

class VoxelDiscriminatorStructureConfigurator(cfg_base.BasicConvNetworkStructureConfigurator):
    def __init__(self):
        super().__init__()

        self.sn = int(0)

        self.nc_f = int(32)
        self.nd_l = int(4)

        self.n_head = int(1)
        self.n_shared = int(1)

class MPGANGeneratorConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.truncation_sigma = float(-1)
        self.network = VoxelGeneratorStructureConfigurator()

class MPGANDiscriminatorConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.network = ImageDiscriminatorStructureConfigurator()

class MPGANCameraConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.fov_degree = float(0)
        self.viewport = list([-1,1,-1,1])

        self.z_near = float(0)
        self.z_far = float(0)
        self.z_sample_cnt = int(64)

        self.image_size = list([32, 32])

class MPGANProjectionConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.view_def_file = str()
        self.cluster_def_file = str()
        self.render_method = str('drc')
        
        self.camera = MPGANCameraConfigurator()

class MPGANLossConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()

        self.gan_loss = str('hinge')
        self.gp = float(0)
        self.tv_voxel = float(-1)
        self.beta_prior = float(-1)

class MPGANSolverConfigurator(cfg_base.BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.param = str()
        self.dg_ratio = int(2)

class MPGANRunnerConfigurator(cfg_base.BasicRunnerConfigurator):
    def __init__(self):
        super().__init__()

        self.data_loader_train = MPGANDataLoaderConfigurator()
        self.data_loader_test = MPGANDataLoaderConfigurator()
        self.generator = MPGANGeneratorConfigurator()
        self.discriminator = MPGANDiscriminatorConfigurator()
        self.projection = MPGANProjectionConfigurator()
        self.loss = MPGANLossConfigurator()
        self.solver_G = MPGANSolverConfigurator()
        self.solver_D = MPGANSolverConfigurator()