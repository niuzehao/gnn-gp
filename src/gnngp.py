import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional
import compute, predict, datasets

class GNNGP(object):
    """
    Build an infinite-width Graph Neural Network, and compute its Gaussian Process Kernel.

    Args:
        data (torch_geometric.data.data): a graph data object.
        device (torch.device): which device used in computation.
        L (int): number of layers of the Graph NN. (default: `2`)
        sigma_b (float): bias variance of the Graph NN. (default: `0.1`)
        sigma_w (float): weight variance of the Graph NN. (default: `1.0`)
        Nystrom (bool): whether use Nystrom approximation in computation. (default: `False`)
        initial (str): specify the initial kernel function used. supported values:
            "linear": linear product kernel $k(x,y)=x^Ty$.
            "rbf": radial basis function kernel $k(x,y)=\exp(-\gamma\Vert x-y\Vert_2^2)$.
            "laplacian": laplacian kernel $k(x,y)=\exp(-\gamma\Vert x-y\Vert_1)$.
            "arccos": arccos kernel $k(x,y)=1-\frac{1}{\pi}\arccos\frac{x^Ty}{\Vert x\Vert_2\Vert y\Vert_2}$.
            "sigmoid": sigmoid kernel $k(x,y)=\tanh(\gamma x^Ty+c)$.
            "polynomial": polynomial kernel $k(x,y)=(x^Ty+c)^d$.
            (default: "linear")
        method (str): specify the architecture used. supported values:
            "GCN": graph convolutional network $X\gets AXW$.
            "GCN2": GCN with initial residual connections and identity mapping (GCNII) $X\gets ((1-\alpha)AX+\sigma X^{(0)})((1-\beta)I+\beta W)$.
            "GIN": graph isomorphism network $X\gets h(AX)$.
            "SAGE": graph sample and aggregate network $X\gets W_1X+W_2AX$.
            "GGP": graph gaussian process kernel (for benchmark use).
            "RBF": radial basis function kernel (for benchmark use).
            (default: "GCN")
        **params (Dict, optional): extra arguments to `initial` and `method`.
            For `initial`, supported arguments:
                gamma (float): to specify the "rbf", "laplacian" and "sigmoid" kernel.
                c (float): to specify the "sigmoid" and "polynomial" kernel.
                d (float): to specify the "polynomial" kernel.
            For `method`, supported arguments:
                alpha (float): to specify the "GCN2" method.
                theta (float): to specify the "GCN2" method $\beta_l=\log(\frac{\theta}{l}+1)$. 

    Attributes:
        N (int): number of nodes of the graph.
        X (Tensor[float]): input features.
        y (Tensor[float] or Tensor[int]): prediction target.
            For int type, the task is a classification problem.
            For float type, the task is a regression problem.
        A (torch.sparse_coo_tensor): normalized graph adjacency matrix.
        mask (Dict[str, Tensor[bool]]): a dictionary containing masks used in computation. necessary keys:
            "train": for training mask.
            "val": for validation mask.
            "test": for test mask.
            "landmark": for Nystrom approximation.
        computed (bool): indicator of availability of following attributes:
            C0 (Tensor[float]): the initial kernel of input features.
            K (Tensor[float]): the GNNGP Kernel.
            Q0 (Tensor[float]): Nystrom approx square root of the initial kernel.
            Q (Tensor[float]): Nystrom approx square root of the GNNGP kernel.
        fit (Tensor[float]): predictions for a range of nugget values.
        result (Dict[str, Tensor[float]]): result metric for a range of nugget values.
        nugget (Tensor[float]): the nugget used in posterior inference.
    """
    def __init__(self, data, device:torch.device=None,
                 L:int=2, sigma_b:float=0.1, sigma_w:float=1.0,
                 Nystrom:bool=False, initial:str="linear", method:str="GCN", **params):
        self.set_hyper_param(L, sigma_b, sigma_w, Nystrom, initial, method, **params)
        self.Nystrom = Nystrom
        self.device = device
        self.N, self.X, self.y, self.A, self.mask = datasets.get_data(data.to(self.device))

    def set_hyper_param(self, L:int=None, sigma_b:float=None, sigma_w:float=None,
                        Nystrom:bool=None, initial:str=None, method:str=None, **params) -> None:
        """
        Set hyper parameters of the Gaussian Process Kernel.

        See `GNNGP` object for details.
        """
        if L is not None: self.L = L
        if sigma_b is not None: self.sigma_b = sigma_b
        if sigma_w is not None: self.sigma_w = sigma_w
        if Nystrom is not None: self.Nystrom = Nystrom
        if initial is not None: self.initial = initial
        if method is not None: self.method = method
        if params is not None: self.params = params
        self.computed = False

    def get_kernel(self) -> Tensor:
        """
        Get Gaussian Process Kernel of the infinite-width Graph Neural Network.
        """
        if not self.computed:
            if self.Nystrom:
                self.Q0 = compute._init_kernel_Nystrom(self.X, self.mask["landmark"], self.initial, **self.params)
                self.Q = compute._get_kernel_Nystrom(self.Q0, self.A, self.L, self.sigma_b, self.sigma_w, self.mask["landmark"], self.method, **self.params)
            else:
                self.C0 = compute._init_kernel(self.X, self.initial, **self.params)
                self.K = compute._get_kernel(self.C0, self.A, self.L, self.sigma_b, self.sigma_w, self.method, **self.params)
            self.computed = True
        return self.Q if self.Nystrom else self.K

    def predict(self, nugget:Union[float, Tensor]=1e-2):
        """
        Make predictions for a range of nugget values in one shot.
        Then compute the train, validation and test result metric.
        
        +----------------+----------------------------+------------------------------+
        | task           | prediction target          | result metric                |
        +----------------+----------------------------+------------------------------+
        | classification | classification probability | mean classification accuracy |
        | regression     | output value               | R-squared statistic          |
        +----------------+----------------------------+------------------------------+

        Args:
            nugget (float or Tensor[float]): the nugget used in posterior inference.
        """
        self.get_kernel()
        if isinstance(nugget, float): nugget = torch.tensor([nugget])
        if self.Nystrom:
            self.fit = predict.fit_Nystrom(self.Q, self.y, self.mask["train"], nugget)
        else:
            self.fit = predict.fit(self.K, self.y, self.mask["train"], nugget)
        self.result = predict.result(self.fit, self.y, self.mask)
        self.nugget = nugget
        return self.fit

    def get_summary(self):
        """
        Returns a dictionary of model hyper-parameters and best train, validation and test results.
        """
        summary = {"L": self.L, "sigma_b": self.sigma_b, "sigma_w": self.sigma_w, "Nystrom": self.Nystrom, "initial": self.initial, "method": self.method}
        if hasattr(self, "result"):
            result = self.result
            i = torch.argmax(result["val"])
            summary.update({"nugget": self.nugget[i], "train": result["train"][i], "val": result["val"][i], "test": result["test"][i]})
        return summary

