import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional
import compute, predict, datasets

class GNNGP(object):
    """
    Build an infinite-width Graph Neural Network model, and compute its Gaussian Process Kernel.

    Args:
        data (torch_geometric.datasets): a graph object.
        L (int): number of hidden layers of the Graph NN. (default: `1`)
        sigma_b (float): bias variance of the Graph NN. (default: `0.1`)
        sigma_w (float): weight variance of the Graph NN. (default: `1.0`)
        Nystrom (bool): whether use Nystrom approximation in computation.
        device (int): which GPU in computation when available. 
    """
    def __init__(self, data, L:int=1, sigma_b:float=0.1, sigma_w:float=1.0,
                 Nystrom:bool=False, device:int=0, **params):
        self.set_hyper_param(L, sigma_b, sigma_w)
        self.Nystrom = Nystrom
        self.device = torch.device('cuda:%s' % device if device>=0 else 'cpu')
        self.N, self.X, self.y, self.A, self.mask = datasets.get_data(data.to(self.device))
    
    def set_hyper_param(self, L:int, sigma_b:float, sigma_w:float) -> None:
        """
        Set hyper parameters of the Gaussian Process Kernel.

        Args:
            L (int): number of hidden layers of the Graph NN.
            sigma_b (float): bias variance of the Graph NN.
            sigma_w (float): weight variance of the Graph NN.
        """
        self.L = L
        self.sigma_b = sigma_b
        self.sigma_w = sigma_w
        self.computed = False

    def set_init_kernel(self, kernel:str="linear", **params) -> None:
        """
        Set the Gaussian Process Kernel of the initial input features.

        Args:
            kernel (string): (default: "linear") supported values:
                "linear": linear product kernel $k(x,y)=x^Ty$.
                "rbf": radial basis function kernel $k(x,y)=\exp(-\gamma\Vert x-y\Vert_2^2)$.
                "laplacian": laplacian kernel $k(x,y)=\exp(-\gamma\Vert x-y\Vert_1)$.
                "arccos": arccos kernel $k(x,y)=1-\frac{1}{\pi}\arccos\frac{x^Ty}{\Vert x\Vert_2\Vert y\Vert_2}$.
                "sigmoid": sigmoid kernel $k(x,y)=\tanh(\gamma x^Ty+c)$.
                "polynomial": polynomial kernel $k(x,y)=(x^Ty+c)^d$.
            **params (dict, optional): extra arguments to `kernel`. supported arguments:
                gamma (float): to specify the "rbf", "laplacian" and "sigmoid" kernel.
                c (float): to specify the "sigmoid" and "polynomial" kernel.
                d (float): to specify the "polynomial" kernel.
        """
        if self.Nystrom:
            self.Q0 = compute._init_kernel_Nystrom(self.X, self.mask["landmark"], kernel, **params)
        else:
            self.K0 = compute._init_kernel(self.X, kernel, **params)

    def add_layer(self, method:str="GCN", **params) -> None:
        """
        Add one more layer to the infinite-width Graph Neural Network, and recompute the Gaussian Process Kernel.

        Args:
            method (string): (default: "GCN") supported values:
                "GCN": Graph Convolutional Network.
            Nystrom (bool): whether use a Nystrom in the computation.
            **params (dict, optional): extra arguments to `method`. supported arguments:
                mask (Tensor): the mask for landmark points in Nystrom approximation.
        """
        if not self.computed:
            raise Exception("Must compute kernel before add another layer!")
        if self.Nystrom:
            self.Q = compute._add_layer_Nystrom(self.Q, self.A, self.sigma_b, self.sigma_w, self.mask["landmark"], method, **params)
        else:
            self.K = compute._add_layer(self.K, self.A, self.sigma_b, self.sigma_w, method, **params)
        self.L += 1

    def get_kernel(self, method:str="GCN", **params) -> Tensor:
        """
        Get Gaussian Process covariance or recompute a new one.

        Args:
            method (string): (default: "GCN") supported values:
                "GCN": Graph Convolutional Network.
            Nystrom (bool): whether use a Nystrom in the computation.
            **params (dict, optional): extra arguments to `method`. supported arguments:
                mask (Tensor): the mask for landmark points in Nystrom approximation.
        """
        if not self.computed:
            self.set_init_kernel(**params)
            if self.Nystrom:
                self.Q = compute._get_kernel_Nystrom(self.Q0, self.A, self.L, self.sigma_b, self.sigma_w, self.mask["landmark"], method, **params)
            else:
                self.K = compute._get_kernel(self.K0, self.A, self.L, self.sigma_b, self.sigma_w, method, **params)
            self.computed = True
        return self.Q if self.Nystrom else self.K

    def get_summary(self):
        """
        Returns a dictionary of model hyper-parameters and best train, validation and test results.
        """
        summary = {"L": self.L, "sigma_b": self.sigma_b, "sigma_w": self.sigma_w}
        result = self.result
        i = torch.argmax(result["val"])
        summary.update({"nugget": self.nugget[i], "train": result["train"][i], "val": result["val"][i], "test": result["test"][i]})
        return summary

    def predict(self, nugget:Union[float,List[float]], **params):
        """
        Making predictions of target using training data.
        For classification problems, the one-hot encoding is applied,
        and the predicted class is the row-wise maximum.

        Args:
            nugget (float or List[float]): the nugget used in posterior inference.
        """
        self.nugget = nugget
        self.get_kernel(**params)
        if self.Nystrom:
            self.fit = predict.fit_Nystrom(self.Q, self.y, self.mask["train"], self.mask["landmark"], nugget)
        else:
            self.fit = predict.fit(self.K, self.y, self.mask["train"], nugget)
        return self.fit

    def get_result(self, nugget:Union[float,List[float]], **params):
        """
        Making predictions of target using training data, and then compute the
        train, validation and test result for each nugget given.
        For classification problems, the result is mean classification accuracy.
        For regression problems, the result is the R-squared statistic.

        Args:
            nugget (float or List[float]): the nugget used in posterior inference.
        """
        self.predict(nugget, **params)
        self.result = predict.result(self.fit, self.y, self.mask)
        return self.result

