# Contains RandomFourier Features from GP Sinkhorn repository.
import math
from copy import deepcopy
from conv_kernel import ConvSimple
import numpy as np
import torch
from model_utils import PositionalEmbedding
from pyro.contrib.gp.kernels import RBF, Exponential, Kernel
from pyro.nn.module import PyroParam
from torch import nn
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import torch.nn.functional as F

# from conv_rff
def extract_image_patches(x, patch_size, stride=1, dilation=1):
    num_windows = x.shape[2] - patch_size + 1
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (num_windows - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (num_windows - 1) * dilation + 1 - w
    # print("padding: " + str(pad_row))
    pad_row, pad_col = 0, 0 # why?
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    # Extract patches
    # patches = x.unfold(2, num_windows, stride).unfold(3, num_windows, stride)
    print("before unfold")
    print(x.shape)
    patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # patches = patches.permute(0,4,5,1,2,3).contiguous()#.cuda()
    print("before reshape")
    print(patches.shape)
    print("b: " + str(b))
    temp = patches.reshape(b,-1,patches.shape[-2], patches.shape[-1])
    print("after reshape")
    print("patches shape: " + str(temp.shape))
    return temp
    # return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

class ArcCos(Kernel):

    def __init__(self, input_dim, variance_w, variance_b, variance=None,
                 lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        self.variance_w = variance_w
        self.variance_w = PyroParam(variance_w, constraints.positive)

        self.variance_b = variance_b
        self.variance_b = PyroParam(variance_b, constraints.positive)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)

    def transform(self, x):
        return self.variance_w * x + self.variance_b

    def forward(self, X, Z=None):
        """ Compute kernel matrix. """
        if Z is None:
            Z = X
        xz = self.transform(X.mm(Z.T))

        X_norm = torch.sqrt(self.transform(torch.linalg.norm(X, dim=1) ** 2))
        Z_norm = torch.sqrt(self.transform(torch.linalg.norm(Z, dim=1) ** 2))

        multiplier = torch.outer(X_norm, Z_norm)

        cos_theta = torch.clip(xz / multiplier, -1, 1)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)

        J = sin_theta + (torch.pi - torch.acos(cos_theta)) * cos_theta

        # TODO: does it matter whether the denominator is pi or 2 * pi?
        # CNN-GP folk use 2 * pi, but pi is also believable and works.
        return self.variance * J * multiplier / (2 * torch.pi)

_SUPPORTED_KERNELS = (Exponential, RBF, ArcCos)

class RandomFourierFeatures(nn.Module):
    """ Implementation for random features approach to exact kernel
        approximation: sample some weights, and use random features to perform
        regression.
        Note that the NN kernel is also supported, although technically its
        random features are not derived using Fourier analysis.
    """

    def __init__(
        self,
        x,
        y,
        t,
        num_features,
        emb_size=128,
        kernel=RBF,
        noise=1,
        device=None,
        random_seed=None,
        debug_rff=False,
        jitter=1e-6,
        sin_cos=False,
        var_w=1,
        var_b=1,
        patch_size = 5,
        stride = 1,
        image_size = 28,
        depth = 1,
        growth_factor = 1,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal"):

        super().__init__() # Added when converting to nn.Module

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.t = t
        self.num_features = num_features
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.growth_factor = growth_factor
        self.image_size = image_size
        self.kernel = self.init_kernel(kernel)
        self.noise = noise + jitter
        self.sin_cos = sin_cos
        self.variance_w = var_w
        self.variance_b = var_b
        self.depth = depth

        # From Hamza
        self.emb_size = emb_size
        self.time_emb = time_emb
        self.input_emb = input_emb

        self.time_mlp = PositionalEmbedding(self.emb_size, self.time_emb)
        self.input_mlps = []
        for i in range(x.shape[1]):
            # input_mlp = PositionalEmbedding(self.emb_size, self.input_emb, scale=25.0, device=self.device)
            input_mlp = PositionalEmbedding(self.emb_size, self.input_emb, scale=25.0)
            self.input_mlps.append(input_mlp)

        # Do the equivalent of a forward pass
        emb_list = []
        for i in range(len(self.input_mlps)):
            input_mlp = self.input_mlps[i]
            x_emb = input_mlp(x[:, i]).float()
            emb_list.append(x_emb)
        t_emb = self.time_mlp(t)
        emb_list.append(t_emb)

        # Reset self.x to be the concatenation of the embeddings
        self.x = torch.cat(emb_list, dim=-1)

        # (1000, 130)

        ################ Naive Implementation #####################
        # self.W = nn.Parameter(torch.Tensor(self.num_features, self.x.shape[-1]).normal_(0, 1), requires_grad=False)

        ############################################################

        ######################### BASE RFF IMPLEMENTATION (Not working) #########################

        self.arccos = isinstance(self.kernel, ArcCos)

        # From conv_rff
        if isinstance(self.kernel, ConvSimple):
            self.conv = True
            self.patch_sizes = self.kernel.patch_sizes
        else:
            self.conv = False

        self.feature_mapping_fn = (
            self.feature_mapping_nn_simple if self.arccos else
            self.feature_mapping_conv if self.conv else # from conv_rff
            self.feature_mapping_dskn if kernel == "DSKN" else
            self.feature_mapping_deep_conv if kernel == "DeepConv" else
            self.feature_mapping_deep_conv_2 if kernel == "DeepConv2" else
            self.feature_mapping_rff)

        if not self.arccos:
            self.f_kernel = self.fourier_transform(self.kernel)

        self.init_params()
        if kernel != "DSKN" and kernel != "DeepConv" and kernel != "DeepConv2":
            self.variance = self.kernel.variance

        self.phi = self.feature_mapping_fn(self.x)
        # self.ws = self.solve_w(self.phi, self.y, self.noise)
        temp = self.solve_w(self.phi, self.y, self.noise)
        self.ws = temp

        print("################ Base RFF #################")
        print(f"Feature shape: {self.phi.shape}")
        #print("Features")
        #print(self.phi)
        print(f"Weights shape: {temp.shape}")
        #print("Weights")
        #print(temp)

        ####################################################################################################################################


    def init_kernel(self, kernel):
        """ Check whether we have an instance of the kernel, and instantiate
            one (with default params) if not.
        """
        if isinstance(kernel, _SUPPORTED_KERNELS):
            kernel = deepcopy(kernel)
        elif isinstance(kernel, Kernel):
            raise NotImplementedError("Unsupported kernel")
        elif kernel == ArcCos:
            kernel = ArcCos(input_dim=self.x.shape[1], variance_w=torch.tensor(1.0), variance_b=torch.tensor(1.0))
        elif kernel == ConvSimple:
            # forward_kernel = RBF(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            kernel = ConvSimple(input_dim=self.x.shape[1], kernel_underlying_forward=forward_kernel, patch_sizes=[self.patch_size])
        elif kernel == "DSKN":
            kernel = "DSKN"
        elif kernel == "DeepConv":
            kernel = "DeepConv"
        elif kernel == "DeepConv2":
            kernel = "DeepConv2"
        else:
            kernel = kernel(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
        return kernel

    def debug_kernel(self):
        kernel_exact = self.kernel.forward(self.x)
        kernel_approx = self.phi @ self.phi.t()
        return kernel_exact, kernel_approx

    def get_f_kernel(self, kernel, dim_x):
        mean = torch.zeros(dim_x)
        if isinstance(kernel, RBF):
            sigma_new = 1 / kernel.lengthscale ** 2
            variance = sigma_new * torch.eye(dim_x)
            return MultivariateNormal(mean, variance).sample

        elif isinstance(kernel, Exponential):
            sigma_new = 1 / kernel.lengthscale
            variance = sigma_new * torch.eye(dim_x)
            def sample_exp(sample_shape):
                gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device),
                                    (dim_x, 1)).T
                gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                return gaussians / torch.sqrt(gammas)
            return sample_exp

        # Set forward kernel for time embedding
        elif isinstance(kernel, ConvSimple):
            dim_x = self.emb_size
            mean = torch.zeros(dim_x)
            if isinstance(kernel.kernel_forwards[0], RBF):
                sigma_new = 1 / kernel.lengthscale ** 2
                variance = sigma_new * torch.eye(dim_x)
                return MultivariateNormal(mean, variance).sample
            elif isinstance(kernel.kernel_forwards[0], Exponential):
                sigma_new = 1 / kernel.lengthscale
                variance = sigma_new * torch.eye(dim_x)
                def sample_exp(sample_shape):
                    gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device),
                                        (dim_x, 1)).T
                    gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                    return gaussians / torch.sqrt(gammas)
                return sample_exp

    def fourier_transform(self, kernel):
        """ Compute the Fourier Transform of the kernel, returning a sampling
            function for the transformed density.
        """
        dim_x = self.x.shape[1]
        mean = torch.zeros(dim_x)
        if isinstance(kernel, RBF):
            sigma_new = 1 / kernel.lengthscale ** 2
            variance = sigma_new * torch.eye(dim_x)
            return MultivariateNormal(mean, variance).sample

        elif isinstance(kernel, Exponential):
            sigma_new = 1 / kernel.lengthscale
            variance = sigma_new * torch.eye(dim_x)
            def sample_exp(sample_shape):
                gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device),
                                    (dim_x, 1)).T
                gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                return gaussians / torch.sqrt(gammas)
            return sample_exp

        # Set forward kernel for time embedding
        elif isinstance(kernel, ConvSimple):
            dim_x = self.emb_size
            mean = torch.zeros(dim_x)
            if isinstance(kernel.kernel_forwards[0], RBF):
                sigma_new = 1 / kernel.lengthscale ** 2
                variance = sigma_new * torch.eye(dim_x)
                return MultivariateNormal(mean, variance).sample
            elif isinstance(kernel.kernel_forwards[0], Exponential):
                sigma_new = 1 / kernel.lengthscale
                variance = sigma_new * torch.eye(dim_x)
                def sample_exp(sample_shape):
                    gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device),
                                        (dim_x, 1)).T
                    gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                    return gaussians / torch.sqrt(gammas)
                return sample_exp

    def init_params(self):
        """ Randomly sample parameters of appropriate shapes to use in later
            feature mapping functions. It is crucial to use the same random
            weights at train and test time, which is why member variables
            are set.
        """
        if self.arccos:
            n, dim_x = self.x.shape

            std_w = math.sqrt(self.variance_w  / dim_x)
            std_b = math.sqrt(self.variance_b)

            self.w0 = torch.normal(0, std_w, size=([dim_x, self.num_features])).double().to(self.device)
            self.b0 = torch.normal(0, std_b, size=([self.num_features])).to(self.device)
        elif self.conv: # from conv_rff
            n, dim_x = self.x.shape
            std_w = math.sqrt(self.variance_w  / dim_x)
            std_b = math.sqrt(self.variance_b)
            print("std_w: " + str(std_w))
            print("std_b: " + str(std_b))

            self.num_x_features = self.num_features
            self.num_t_features = 500

            # for phi_t
            forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            f_kernel = self.get_f_kernel(forward_kernel, 128)
            self.omega = self.f_kernel([self.num_t_features]).double().to(self.device).t()
            self.b = Uniform(0, 2 * math.pi).sample([self.num_t_features]).to(self.device)

            self.conv_ws = []
            self.conv_bs = []
            for patch_size in self.patch_sizes:
                num_patches = ((self.image_size - self.patch_size + 1) // self.stride) ** 2
                len_input = num_patches * self.patch_size ** 2
                # self.conv_ws.append(torch.normal(0, std_w, size=([len_input, self.num_features])).double().to(self.device))
                # self.conv_ws.append(torch.normal(0, std_w, size=([len_input + 128, self.num_features])).double().to(self.device))
                self.conv_ws.append(torch.normal(0, std_w, size=([len_input, self.num_x_features])).double().to(self.device)) # separating phi_x and phi_t
                self.conv_bs.append(torch.normal(0, std_b, size=([self.num_x_features])).to(self.device))

        elif self.kernel == "DSKN": # From DSKN

            ############################### Approach 1 ########################

            # forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            # num_x_features = 3000
            # self.num_x_features = num_x_features
            # num_t_features = 3000
            # self.num_t_features = num_t_features
            # f_kernel = self.get_f_kernel(forward_kernel, 2)
            # self.x_omegas = [f_kernel([num_x_features]).double().to(self.device).t()]
            # self.x_bs = [Uniform(0, 2 * math.pi).sample([num_x_features]).to(self.device)]
            # f_kernel = self.get_f_kernel(forward_kernel, 128)
            # self.t_omegas = [f_kernel([num_t_features]).double().to(self.device).t()]
            # self.t_bs = [Uniform(0, 2 * math.pi).sample([num_t_features]).to(self.device)]
            # for i in range(1, self.depth):
            #     if self.sin_cos:
            #         temp_kernel = self.get_f_kernel(forward_kernel, num_x_features*2)
            #         x_omega = temp_kernel([num_x_features]).double().to(self.device).t()
            #         temp_kernel = self.get_f_kernel(forward_kernel, num_t_features*2)
            #         t_omega = temp_kernel([num_t_features]).double().to(self.device).t()
            #     else:
            #         temp_kernel = self.get_f_kernel(forward_kernel, num_x_features)
            #         x_omega = temp_kernel([num_x_features]).double().to(self.device).t()
            #         temp_kernel = self.get_f_kernel(forward_kernel, num_t_features)
            #         t_omega = temp_kernel([num_t_features]).double().to(self.device).t()
            #     x_b = Uniform(0, 2 * math.pi).sample([num_x_features]).to(self.device) # change if not same number of features in each layer!
            #     t_b = Uniform(0, 2 * math.pi).sample([num_t_features]).to(self.device)
            #     self.x_omegas.append(x_omega)
            #     self.t_omegas.append(t_omega)
            #     self.x_bs.append(x_b)
            #     self.t_bs.append(t_b)

            #####################################################################

            # for i in range(1, self.depth):
            #     if self.sin_cos:
            #         temp_kernel = self.get_f_kernel(forward_kernel, self.num_features*2)
            #         omega = temp_kernel([self.num_features]).double().to(self.device).t()
            #     else:
            #         temp_kernel = self.get_f_kernel(forward_kernel, self.num_features)
            #         omega = temp_kernel([self.num_features]).double().to(self.device).t()
            #     b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device) # change if not same number of features in each layer!
            #     self.omegas.append(omega)
            #     self.bs.append(b)

            # self.omega = f_kernel([self.num_features]).double().to(self.device).t()
            # self.b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device)

            ########################## Approach 2 ###############################

            self.num_x_features = 3000
            self.num_t_features = 3000
            self.sigma0 = 2 #1e-2
            self.sigma1 = 2 #1e-6
            # self.growth_factor = 1

            # For deep RFF for x  (based on DSKN)
            self.x_omegas_1 = [Normal(0, self.sigma0).sample([2, self.num_x_features]).double().to(self.device)]
            self.x_omegas_2 = [Normal(0, self.sigma1).sample([2, self.num_x_features]).double().to(self.device)]
            self.x_bs_1 = [Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device)]
            self.x_bs_2 = [Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device)]
            for i in range(1, self.depth):
                self.x_omegas_1.append(Normal(0, self.sigma0 * math.pow(self.growth_factor, i)).sample([self.num_x_features, self.num_x_features]).double().to(self.device))
                self.x_omegas_2.append(Normal(0, self.sigma1 * math.pow(self.growth_factor, i)).sample([self.num_x_features, self.num_x_features]).double().to(self.device))
                self.x_bs_1.append(Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device))
                self.x_bs_2.append(Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device))

            # For shallow time RFF
            forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            f_kernel = self.get_f_kernel(forward_kernel, 128)
            self.t_omegas = [f_kernel([self.num_t_features]).double().to(self.device).t()]
            self.t_bs = [Uniform(0, 2 * math.pi).sample([self.num_t_features]).to(self.device)]


            ######################## Original DSKN #################################

            # self.num_dimension = self.x.shape[1]
            # self.out_dimension = [self.num_features for i in range(self.depth)]
            # self.sigma0 = 1e-2
            # self.sigma1 = 1e-6
            # self.growth_factor = 1
            #
            # self.op_list0 = [nn.Linear(self.num_dimension, self.out_dimension[0])]
            # self.op_list1 = [nn.Linear(self.num_dimension, self.out_dimension[0])]
            # for i in range(1, len(self.out_dimension)):
            #     op_tmp0 = nn.Linear(self.out_dimension[i-1], self.out_dimension[i])
            #     op_tmp1 = nn.Linear(self.out_dimension[i-1], self.out_dimension[i])
            #     self.op_list0.append(op_tmp0)
            #     self.op_list1.append(op_tmp1)
            # self.op_list0 = nn.ModuleList(self.op_list0)
            # self.op_list1 = nn.ModuleList(self.op_list1)
            #
            # for i in range(len(self.op_list0)):
            #     nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
            #     nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
            #     nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
            #     self.op_list0[i].weight.requires_grad = False
            #     self.op_list1[i].weight.requires_grad = False
            #     self.op_list0[i].bias.requires_grad = False
            #     self.op_list1[i].bias = self.op_list0[i].bias

            #####################################################################
        elif self.kernel == "DeepConv":
            n, dim_x = self.x.shape
            std_w = math.sqrt(self.variance_w  / dim_x)
            std_b = math.sqrt(self.variance_b)

            self.num_x_features = self.num_features
            self.num_t_features = 500

            self.sigma0 = 0.03#1e-3 #1e-2
            self.sigma1 = 0.03#1e-3 #1e-6
            self.growth_factor = 1.0

            self.kernel_size = 5
            out_channels = 1 # Not sure what to use here
            _padding = 'same' #0 # default padding
            padding_mode = 'zeros'
            self.out_dimension = out_channels * self.image_size * self.image_size # not sure here

            # For deep RFF for x (based on CSKN)
            self.op_list0 = [nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=self.kernel_size, padding=_padding, padding_mode=padding_mode)]
            self.op_list1 = [nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=self.kernel_size, padding=_padding, padding_mode=padding_mode)]
            for _ in range(self.depth - 1):
                op_tmp0 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=_padding, padding_mode=padding_mode)
                op_tmp1 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=_padding, padding_mode=padding_mode)
                self.op_list0.append(op_tmp0)
                self.op_list1.append(op_tmp1)
            self.op_list0 = nn.ModuleList(self.op_list0)
            self.op_list1 = nn.ModuleList(self.op_list1)
            for i in range(len(self.op_list0)):
                nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
                nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
                nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
                self.op_list0[i].weight.requires_grad = False
                self.op_list1[i].weight.requires_grad = False
                self.op_list0[i].bias.requires_grad = False
                self.op_list1[i].bias = self.op_list0[i].bias

            # For deep RFF for x  (based on DSKN)
            # num_patches = ((self.image_size - self.patch_size + 1) // self.stride) ** 2
            # len_input = num_patches * self.patch_size ** 2
            # # self.x_omegas_1 = [torch.normal(0, std_w, size=([len_input, self.num_x_features])).double().to(self.device)] # Arjun
            # # self.x_omegas_2 = [torch.normal(0, std_w, size=([len_input, self.num_x_features])).double().to(self.device)] # Arjun
            # self.x_omegas_1 = [Normal(0, self.sigma0).sample([len_input, self.num_x_features]).double().to(self.device)] # DSKN
            # self.x_omegas_2 = [Normal(0, self.sigma1).sample([len_input, self.num_x_features]).double().to(self.device)] # DSKN
            # self.x_bs_1 = [Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device)]
            # self.x_bs_2 = [Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device)]
            # for i in range(1, self.depth):
            #     self.x_omegas_1.append(Normal(0, self.sigma0 * math.pow(self.growth_factor, i)).sample([len_input, self.num_x_features]).double().to(self.device))
            #     self.x_omegas_2.append(Normal(0, self.sigma1 * math.pow(self.growth_factor, i)).sample([len_input, self.num_x_features]).double().to(self.device))
            #     self.x_bs_1.append(Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device))
            #     self.x_bs_2.append(Uniform(0, 2 * math.pi).sample([self.num_x_features]).to(self.device))

            # For shallow time RFF
            forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            f_kernel = self.get_f_kernel(forward_kernel, self.emb_size)
            self.t_omega = f_kernel([self.num_t_features]).double().to(self.device).t()
            self.t_b = Uniform(0, 2 * math.pi).sample([self.num_t_features]).to(self.device)

        elif self.kernel == "DeepConv2":

            # Settings
            # n, dim_x = self.x.shape
            # std_w = math.sqrt(self.variance_w  / dim_x)
            # std_b = math.sqrt(self.variance_b)
            # self.num_x_features = self.num_features
            self.num_t_features = 500
            self.sigma0 = 1e-7
            self.sigma1 = 1e-7
            self.growth_factor = 1.0
            self.out_list =  [8, 16, 16, 16, 16, 16, 16] #[64, 64, 128, 128, 256, 256, 512]
            self.pool_dict = {0, 2, 4, 6}
            in_channels = 1
            kernel_size = 3

            # For deep RFF for x (based on CSKN8)
            self.op_list0 = [nn.Conv2d(in_channels, self.out_list[0], kernel_size, padding=1)]
            self.op_list1 = [nn.Conv2d(in_channels, self.out_list[0], kernel_size, padding=1)]
            self.bn_list = [nn.BatchNorm2d(self.out_list[0], affine=False)]
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
            for i in range(1, len(self.out_list)):
                op_tmp0 = nn.Conv2d(self.out_list[i-1], self.out_list[i], kernel_size, padding=1)
                op_tmp1 = nn.Conv2d(self.out_list[i-1], self.out_list[i], kernel_size, padding=1)
                self.op_list0.append(op_tmp0)
                self.op_list1.append(op_tmp1)
                self.bn_list.append(nn.BatchNorm2d(self.out_list[i], affine=False))
            self.op_list0 = nn.ModuleList(self.op_list0)
            self.op_list1 = nn.ModuleList(self.op_list1)
            self.bn_list = nn.ModuleList(self.bn_list)
            for i in range(len(self.op_list0)):
                nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
                nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
                nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
                self.op_list0[i].weight.requires_grad = False
                self.op_list1[i].weight.requires_grad = False
                self.op_list0[i].bias.requires_grad = False
                self.op_list1[i].bias = self.op_list0[i].bias

            # For shallow time RFF
            forward_kernel = Exponential(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
            f_kernel = self.get_f_kernel(forward_kernel, self.emb_size)
            self.t_omega = f_kernel([self.num_t_features]).double().to(self.device).t()
            self.t_b = Uniform(0, 2 * math.pi).sample([self.num_t_features]).to(self.device)

        else:
            # Single-layer RFF
            # self.omega = self.f_kernel([self.num_features]).double().to(self.device).t()
            # self.b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device)
            # Multi-layer RFF
            self.omegas = []
            self.bs = []
            for i in range(self.depth):
                if i == 0:
                    omega = self.f_kernel([self.num_features]).double().to(self.device).t()
                elif self.sin_cos:
                    temp_kernel = self.get_f_kernel(self.kernel, self.num_features*2)
                    omega = temp_kernel([self.num_features]).double().to(self.device).t()
                else:
                    temp_kernel = self.get_f_kernel(self.kernel, self.num_features)
                    omega = temp_kernel([self.num_features]).double().to(self.device).t()
                b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device) # change if not same number of features in each layer!
                self.omegas.append(omega)
                self.bs.append(b)

    def feature_mapping_dskn(self, x):
        # Splitting time and image
        x_part = x[:, :2]
        t_part = x[:, 2:]

        ################### Approach 1 ########################################
        # Based on shallow RFF
        #
        # for i in range(self.depth):
        #     scaling = math.sqrt(2 / self.num_x_features)
        #     basis = x_part.mm(self.x_omegas[i].to(torch.float32)) + self.x_bs[i]
        #     sin_features = torch.sin(basis)
        #     cos_features = torch.cos(basis)
        #     x_part = torch.concat([cos_features, sin_features], axis=1) * scaling
        #
        # # for i in range(self.depth):
        # for i in range(1):
        #     scaling = math.sqrt(2 / self.num_t_features) #* torch.sqrt(self.variance)
        #     basis = t_part.mm(self.t_omegas[i].to(torch.float32)) + self.t_bs[i]
        #     sin_features = torch.sin(basis)
        #     cos_features = torch.cos(basis)
        #     t_part = torch.concat([cos_features, sin_features], axis=1) * scaling
        #
        # return torch.concat([x_part, t_part], axis=1)

        ################################## Approach 2 ###################################
        # Based on DSKN + shallow RFF for time

        # Multi-layer RFF for x
        for i in range(self.depth):
            scaling = 1 / np.sqrt(2 * self.num_x_features)
            basis_1 = x_part.mm(self.x_omegas_1[i].to(torch.float32)) + self.x_bs_1[i]
            features_1 = torch.cos(basis_1)
            basis_2 = x_part.mm(self.x_omegas_2[i].to(torch.float32)) + self.x_bs_2[i]
            features_2 = torch.cos(basis_2)
            x_part = scaling * (features_1 + features_2) # try with concat as well?

        # Shallow RFF for time
        scaling = math.sqrt(2 / self.num_t_features)
        basis = t_part.mm(self.t_omegas[0].to(torch.float32)) + self.t_bs[0]
        sin_features = torch.sin(basis)
        cos_features = torch.cos(basis)
        t_part = torch.concat([cos_features, sin_features], axis=1) * scaling

        return torch.concat([x_part, t_part], axis=1)

        ########################################################################

        # for i in range(self.depth):
        #     scaling = math.sqrt(2 / self.num_features) #* torch.sqrt(self.variance)
        #     basis = x.mm(self.omegas[i].to(torch.float32)) + self.bs[i]
        #     sin_features = torch.sin(basis)
        #     cos_features = torch.cos(basis)
        #     x = torch.concat([cos_features, sin_features], axis=1) * scaling
        # return x

        ##################### Original DSKN ####################################

        # for i in range(len(self.op_list0)):
        #     x = 1 / np.sqrt(2 * self.out_dimension[i]) * (torch.cos(self.op_list0[i](x)) + torch.cos(self.op_list1[i](x)))
        # return x

        ########################################################################

    def feature_mapping_rff(self, x):
        """ Map input x into feature space using params b and omega. """
        scaling = math.sqrt(2 / self.num_features) * torch.sqrt(self.variance)

        # Single layer RFF
        basis = x.mm(self.omega.to(torch.float32)) + self.b
        if self.sin_cos:
            sin_features = torch.sin(basis)
            cos_features = torch.cos(basis)
            return torch.concat([cos_features, sin_features], axis=1) * scaling
        else:
            return torch.cos(basis) * scaling

        # Multi-layer RFF
        # z = x
        # for i in range(self.depth):
        #     w = self.omegas[i]
        #     b = self.bs[i]
        #     z = z.mm(w.to(torch.float32)) + b
        #     if self.sin_cos:
        #         sin_features = torch.sin(z)
        #         cos_features = torch.cos(z)
        #         z = torch.concat([cos_features, sin_features], axis=1) * scaling
        #     else:
        #         z = torch.cos(z) * scaling
        # return z

    def solve_w(self, phi, y, lambda_=0):
        """ Return the weights minimising MSE for basis functions phi and targets
            y, with regularisation coef lambda_.
            Same as torch.linalg.lstsq or solve, except that there's more
            flexible regularisation.
        """
        return (phi.t().mm(phi) +
                torch.rand(phi.size()[-1])/100 * torch.eye(phi.size()[-1], device=self.device)
                ).inverse().mm(phi.t()).mm(y)

    def predict_gp(self, x_pred):
        """ Use the full GP equation to predict the value
            for y_pred, performing the regression once per dimension.
        """
        total = torch.zeros([x_pred.shape[0], self.y.shape[1]], device=self.device)
        phi_pred = self.feature_mapping_fn(x_pred)

        for i in range(self.y.shape[1]):

            pred = phi_pred @ self.phi.t() @ (self.phi @ self.phi.t() +
                                              self.noise *
                                              torch.eye(self.phi.shape[0],
                                                        device=self.device)
                                            ).inverse() @ self.y[:, i]
            total[:, i] = torch.squeeze(pred)
        return total

    def predict(self, x_pred):
        """ Use the object's weights w and input x_pred to predict the value
            for y_pred, performing the regression once per dimension.
        """
        return self.feature_mapping_fn(x_pred).mm(self.ws)

    def forward(self, x, t):
        emb_list = []
        for i in range(len(self.input_mlps)):
            input_mlp = self.input_mlps[i]
            x_emb = input_mlp(x[:, i]).float()
            emb_list.append(x_emb)
        t_emb = self.time_mlp(t)
        emb_list.append(t_emb)
        x = torch.cat(emb_list, dim=-1)

        z = self.feature_mapping_fn(x)

        x = z.mm(self.ws) # z @ w
        return x

    def feature_mapping_nn(self, x, s=1, kappa=1e-6):
        """ Random feature mapping for NN kernel, using the formula from
            Scetbon et. al. (2020).
            TODO: this is not actually used (as we are still trying to
            debug the simple version.
        """
        n, dim_x = x.shape
        variance = self.variance
        C = (variance ** (dim_x / 2)) * np.sqrt(2)
        U = MultivariateNormal(torch.zeros(dim_x), variance * torch.eye(dim_x)
            ).sample([self.num_features]).to(self.device).t().double()

        IP = x.mm(U)

        res_trans = C * (torch.maximum(IP, torch.tensor(0)) ** s)

        V = (variance - 1) / variance
        V = -(1 / 4) * V * torch.sum(U ** 2, axis=0)
        V = torch.exp(V)

        res = torch.zeros((n, self.num_features + 1), dtype=float)

        res[:, :self.num_features] = (1 / math.sqrt(self.num_features)) * res_trans * V
        res[:, -1] = kappa

        scaling = math.sqrt(2 / self.num_features) * torch.sqrt(variance)
        return res.to(self.device) * scaling

    def feature_mapping_nn_simple(self, x):
        """ Random feature mapping for the NN kernel, using the basic random
            feature formula (that in Cho and Saul (2009)). Take the product
            of w with x, add bias b, and use the ReLU nonlearity.
        """
        x1 = torch.maximum(x.mm(self.w0.to(torch.float32)) + self.b0, torch.tensor(0))

        # Scaling1 is the correct one; so we don't normalise by num_features
        scaling1 = torch.sqrt(self.variance) * math.sqrt(2)
        # scaling2 = torch.sqrt(self.variance) * math.sqrt(1 / self.num_features)
        # scaling3 = torch.sqrt(self.variance) * math.sqrt(2 / self.num_features)
        # scaling4 = torch.sqrt(self.variance) * math.sqrt(4 / self.num_features)

        return scaling1 * x1

    @staticmethod
    def flatten(patches):
        return patches.flatten(start_dim=1)

    def feature_mapping_deep_conv(self, x):
        # Separate x and time
        splitting_point = self.image_size * self.image_size
        image_x = x[:,:splitting_point]
        image_x = image_x.unflatten(1, (1, self.image_size, self.image_size)) # reshape as an image
        time_x = x[:,splitting_point:]

        # CSKN
        for i in range(self.depth):
            image_x = 1 / np.sqrt(2 * self.out_dimension) * (torch.cos(self.op_list0[i](image_x)) + torch.sin(self.op_list1[i](image_x)))
            # image_x = 1 / np.sqrt(2 * self.out_dimension) * (torch.cos(self.op_list0[i](image_x)) + torch.cos(self.op_list1[i](image_x)))
            # image_x = 1 / np.sqrt(2 * self.out_dimension) * self.op_list0[i](image_x)
        phi_x = torch.flatten(image_x, 1)

        # Multi-layer RFF for x
        # for i in range(self.depth):
        #     # test
        #     print("Layer: " + str(i))
        #     # Scaling
        #     scaling = 1 / np.sqrt(2 * self.num_x_features) # DSKN scaling
        #     # scaling = torch.sqrt(self.variance) * math.sqrt(2) / self.num_x_features # Arjun conv scaling
        #     # Extract image patches
        #     print("before patches")
        #     print(image_x.shape)
        #     print(image_x)
        #     patches = self.flatten(extract_image_patches(image_x.unflatten(1, (1, self.image_size, self.image_size)), self.patch_size, stride=self.stride))
        #     # Basis
        #     print("before basis")
        #     print(patches.shape)
        #     print(patches)
        #     basis = patches.mm(self.x_omegas_1[i].to(torch.float32)) + self.x_bs_1[i]
        #     # basis_1 = patches.mm(self.x_omegas_1[i].to(torch.float32)) + self.x_bs_1[i]
        #     # basis_2 = patches.mm(self.x_omegas_2[i].to(torch.float32)) + self.x_bs_2[i]
        #     # Features
        #     # features = torch.cos(basis)
        #     # features_1 = torch.cos(basis_1)
        #     # features_2 = torch.cos(basis_2)
        #     # Combining features
        #     print("before scaling")
        #     image_x = scaling * (basis)
        #     # image_x = scaling * (features_1 + features_2)
        # phi_x = image_x

        # Shallow RFF for time
        scaling = math.sqrt(2 / self.num_t_features)
        basis = time_x.mm(self.t_omega.to(torch.float32)) + self.t_b
        sin_features = torch.sin(basis)
        cos_features = torch.cos(basis)
        phi_t = torch.concat([cos_features, sin_features], axis=1) * scaling

        return torch.concat([phi_x, phi_t], axis=1)

    def feature_mapping_deep_conv_2(self, x):
        # Separate x and time
        splitting_point = self.image_size * self.image_size
        image_x = x[:,:splitting_point]
        image_x = image_x.unflatten(1, (1, self.image_size, self.image_size)) # reshape as an image
        time_x = x[:,splitting_point:]

        print("number of layers: " + str(len(self.op_list0)))

        # CSKN8
        for i in range(len(self.op_list0)):
            image_x =  torch.cos(self.op_list0[i](image_x)) + torch.cos(self.op_list1[i](image_x))
            image_x = self.bn_list[i](image_x)
            if i in self.pool_dict:
                image_x = self.max_pool(image_x)
        image_x = self.avg_pool(image_x)
        phi_x = torch.flatten(image_x, 1)

        # Shallow RFF for time
        scaling = math.sqrt(2 / self.num_t_features)
        basis = time_x.mm(self.t_omega.to(torch.float32)) + self.t_b
        sin_features = torch.sin(basis)
        cos_features = torch.cos(basis)
        phi_t = torch.concat([cos_features, sin_features], axis=1) * scaling

        return torch.concat([phi_x, phi_t], axis=1)


    def feature_mapping_conv(self, x):
        # Separate image and time
        splitting_point = self.image_size * self.image_size
        image_x = x[:,:splitting_point]
        time_x = x[:,splitting_point:]
        # Get w and b
        w = self.conv_ws[0]
        b = self.conv_bs[0]
        # Extract image patches
        x_point_patches = self.flatten(extract_image_patches(image_x.unflatten(1, (1, 28, 28)), self.patch_size, stride=self.stride))
        # Concat time embedding
        # x_point_patches = torch.cat((x_point_patches, time_x), dim=-1)
        # x1 = x_point_patches.flatten(start_dim=1).mm(w.float()) + b

        x1 = x_point_patches.mm(w.float()) + b

        # sin_features_x = torch.sin(x1)
        # cos_features_x = torch.cos(x1)

        # sin_features_x = (x1)
        # cos_features_x = (x1)
        # Scale
        scaling = torch.sqrt(self.variance) * math.sqrt(2) / self.num_x_features
        phi_x = scaling *  x1#torch.concat([cos_features_x, sin_features_x], axis=1)

        # RFF for time
        # phi_t = self.feature_mapping_rff(time_x)

        # Shallow RFF for time
        scaling = math.sqrt(2 / self.num_t_features)
        basis = time_x.mm(self.omega.to(torch.float32)) + self.b
        sin_features = torch.sin(basis)
        cos_features = torch.cos(basis)
        phi_t = torch.concat([cos_features, sin_features], axis=1) * scaling

        phi = torch.cat((phi_x, phi_t), dim=-1)
        return phi
        # return scaling * x1
