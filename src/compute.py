import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional
import numpy as np

def _sqrt_Nystrom(K:Tensor, mask:Tensor) -> Tensor:
    """
    Compute the Nystrom approx square root `Ny(K)` of a kernel matrix.
    A smooth function is used for near-zero eigenvalues.

    Args:
        K (Tensor[float]): the `N * Na` partial elements of the kernel matrix.
        mask (Tensor[bool]): the mask for landmark points.
    """
    D, V = torch.linalg.eigh(K[mask])
    D_sqrt = torch.sqrt(torch.clamp(D, 0))
    D_sqrt_inv = D_sqrt / torch.clamp(D, D[-1] * 1e-4)
    Q = torch.zeros_like(K)
    Q[mask] = V * D_sqrt.view((1, -1))
    Q[~mask] = K[~mask] @ V * D_sqrt_inv.view((1, -1))
    return Q


def _init_kernel(X:Tensor, kernel:str, **params) -> Tensor:
    """
    Compute the initial kernel of input features `C0`.

    Args:
        kernel (str): specify the kernel function used (default: "linear"). supported values:
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
    if kernel == "linear":
        C0 = X @ X.T
    elif kernel == "rbf":
        C0 = torch.exp(-params["gamma"]*torch.cdist(X, X, p=2)**2)
    elif kernel == "laplacian":
        C0 = torch.exp(-params["gamma"]*torch.cdist(X, X, p=1))
    elif kernel == "arccos":
        C0 = 1 - torch.arccos(torch.corrcoef(X))/np.pi
    elif kernel == "sigmoid":
        C0 = torch.tanh(params["gamma"] * X @ X.T + params["c"])
    elif kernel == "polynomial":
        C0 = (X @ X.T + params["c"])**params["d"]
    else:
        raise Exception("Unsupported kernel function!")
    return C0


def _init_kernel_Nystrom(X:Tensor, mask:Tensor, kernel:str, **params) -> Tensor:
    """
    Compute Nystrom approx square root `Q0` of the initial kernel `C0`.

    Args:
        kernel (str): specify the kernel function used (default: "linear"). supported values:
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
    if kernel == "linear":
        if torch.sum(mask) >= X.shape[1]: Q0 = X
        else: Q0 = _sqrt_Nystrom(X @ X[mask].T, mask)
    elif kernel == "rbf":
        C0 = torch.exp(-params["gamma"]*torch.cdist(X, X[mask], p=2)**2)
        Q0 = _sqrt_Nystrom(C0, mask)
    elif kernel == "laplacian":
        C0 = torch.exp(-params["gamma"]*torch.cdist(X, X[mask], p=1))
        Q0 = _sqrt_Nystrom(C0, mask)
    elif kernel == "arccos":
        Q0 = X - torch.mean(X, dim=0, keepdim=True)
        Q0 /= torch.sqrt(torch.sum(Q0**2, dim=0, keepdim=True))
        C0 = 1 - torch.arccos(torch.clamp(Q0 @ Q0[mask].T, -1, 1))/np.pi
        Q0 = _sqrt_Nystrom(C0, mask)
    elif kernel == "sigmoid":
        C0 = torch.tanh(params["gamma"] * X @ X[mask].T + params["c"])
        Q0 = _sqrt_Nystrom(C0, mask)
    elif kernel == "polynomial":
        C0 = (X @ X[mask].T + params["c"])**params["d"]
        Q0 = _sqrt_Nystrom(C0, mask)
    else:
        raise Exception("Unsupported kernel function!")
    return Q0


def _get_kernel(C0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, method:str, **params) -> Tensor:
    """
    Compute the GNNGP Kernel `K`.

    Args:
        C0 (Tensor[float]): the `N * N` initial kernel matrix.
        A (Tensor[float]): the `N * N` adjacency matrix.
        L (int): number of layers of the Graph NN.
        sigma_b (float): bias variance of the Graph NN.
        sigma_w (float): weight variance of the Graph NN.
        method (str): specify the architecture used. supported values:
            "GCN": graph convolutional network.
            "GCN2": GCN with initial residual connections and identity mapping (GCNII).
            "GIN": graph isomorphism network.
            "SAGE": graph sample and aggregate network.
            "GGP": graph Gaussian process.
        **params (dict, optional): extra arguments to `method`.
    """
    if method == "GCN":
        return _GCN_kernel(C0, A, L, sigma_b, sigma_w)
    elif method == "GCN2":
        return _GCN2_kernel(C0, A, L, sigma_b, sigma_w, **params)
    elif method == "GIN":
        return _GIN_kernel(C0, A, L, sigma_b, sigma_w, **params)
    elif method == "SAGE":
        return _SAGE_kernel(C0, A, L, sigma_b, sigma_w)
    elif method == "GGP":
        return _GGP_kernel(C0, A)
    else:
        raise Exception("Unsupported layer type!")


def _get_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, method:str, **params) -> Tensor:
    """
    Compute the Nystrom approx square root `Q` of the GNNGP kernel `K`.

    Args:
        Q0 (Tensor[float]): the `N * Na` square root of the initial kernel matrix.
        A (Tensor[float]): the `N * N` adjacency matrix.
        L (int): number of layers of the Graph NN.
        sigma_b (float): bias variance of the Graph NN.
        sigma_w (float): weight variance of the Graph NN.
        mask (Tensor[bool]): the mask for landmark points.
        method (str): specify the architecture used. supported values:
            "GCN": graph convolutional network.
            "GCN2": GCN with initial residual connections and identity mapping (GCNII).
            "GIN": graph isomorphism network.
            "SAGE": graph sample and aggregate network.
            "GGP": graph Gaussian process.
        **params (dict, optional): extra arguments to `method`.
    """
    if method == "GCN":
        return _GCN_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask)
    elif method == "GCN2":
        return _GCN2_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask, **params)
    elif method == "GIN":
        return _GIN_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask, **params)
    elif method == "SAGE":
        return _SAGE_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask)
    elif method == "GGP":
        return _GGP_kernel_Nystrom(Q0, A, mask)
    else:
        raise Exception("Unsupported layer type!")


def _ExxT_ReLU(K:Tensor) -> Tensor:
    """
    Compute `g(K)` for ReLU activation using the closed-form integral result.
    """
    s = torch.sqrt(torch.diag(K)).view((-1, 1))
    theta = torch.arccos(torch.clamp(K/s/s.T, -1, 1))
    ExxT = 0.5/np.pi*(torch.sin(theta)+(np.pi-theta)*torch.cos(theta))*s*s.T
    return ExxT


def _ExxT_ReLU_Nystrom(Q:Tensor, mask:Tensor) -> Tensor:
    """
    Compute `Ny(g(QQ^T))` for ReLU activation using the closed-form integral result.
    """
    s = torch.sqrt(torch.sum(Q**2, 1)).view((-1, 1))
    theta = torch.arccos(torch.clamp(Q @ Q[mask].T/s/s[mask].T, -1, 1))
    ExxT = 0.5/np.pi*(torch.sin(theta)+(np.pi-theta)*torch.cos(theta))*s*s[mask].T
    return _sqrt_Nystrom(ExxT, mask)


def _GCN_kernel(C0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float) -> Tensor:
    ExxT = torch.zeros_like(C0)
    K = sigma_b**2 + sigma_w**2 * A @ (A @ C0).T
    for j in range(L-1):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = sigma_b**2 + sigma_w**2 * A @ (A @ ExxT).T
    return K


def _GCN_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor) -> Tensor:
    N, Ni = Q0.shape; Na = torch.sum(mask)
    Q = torch.zeros((N, Na+1), device=Q0.device)
    Q[:,:Q0.shape[1]] = sigma_w * A @ Q0 ; Q[:,-1] = sigma_b
    for j in range(L-1):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:-1] = sigma_w * A @ ExxT ; Q[:,-1] = sigma_b
    return Q


def _GCN2_kernel(C0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, alpha:float=0.1, theta:float=0.5) -> Tensor:
    ExxT = _ExxT_ReLU(C0)
    K = sigma_w**2 * A @ (A @ ExxT).T
    for j in range(L-1):
        ExxT[:] = _ExxT_ReLU(K)
        beta = np.log(theta/(j+1)+1)
        coef = (sigma_w*beta)**2+(1-beta)**2
        K[:] = coef * ((1-alpha)**2 * A @ (A @ (ExxT)).T + (alpha)**2 * C0)
    return K


def _GCN2_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, alpha:float=0.1, theta:float=0.5) -> Tensor:
    N, Ni = Q0.shape; Na = torch.sum(mask)
    ExxT = _ExxT_ReLU_Nystrom(Q0, mask)
    Q = torch.zeros((N, Na+Ni), device=Q0.device)
    Q[:,:Na] = sigma_w * A @ ExxT
    for j in range(L-1):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        beta = np.log(theta/(j+1)+1)
        coef = np.sqrt((sigma_w*beta)**2+(1-beta)**2)
        Q[:,:Na] = (1-alpha)*coef * A @ ExxT
        Q[:,Na:Na+Ni] = (alpha)*coef * Q0
    return Q


def _GIN_kernel(C0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, eps:float=0.0) -> Tensor:
    ExxT = torch.zeros_like(C0)
    K = sigma_b**2 + sigma_w**2 * A @ (A @ C0).T
    for j in range(L-1):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = sigma_b**2 + sigma_w**2 * ExxT
    return K


def _GIN_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, eps:float=0.0) -> Tensor:
    N, Ni = Q0.shape; Na = torch.sum(mask)
    Q = torch.zeros((N, Na+1), device=Q0.device)
    Q[:,:Ni] = sigma_w * A @ Q0 ; Q[:,-1] = sigma_b
    for j in range(L-1):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:-1] = sigma_w * ExxT ; Q[:,-1] = sigma_b
    return Q


def _SAGE_kernel(C0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float) -> Tensor:
    ExxT = torch.zeros_like(C0)
    K = sigma_b**2 * C0 + sigma_w**2 * A @ (A @ C0).T
    for j in range(L-1):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = sigma_b**2 * ExxT + sigma_w**2 * A @ (A @ ExxT).T
    return K


def _SAGE_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor) -> Tensor:
    N, Ni = Q0.shape; Na = torch.sum(mask)
    Q = torch.zeros((N, 2*Na), device=Q0.device)
    Q[:,:Ni] = sigma_b * Q0; Q[:,Ni:2*Ni] = sigma_w * A @ Q0
    for j in range(L-1):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:Na] = sigma_b * ExxT
        Q[:,Na:] = sigma_w * A @ ExxT
    return Q


def _GGP_kernel(C0:Tensor, A:Tensor) -> Tensor:
    K = A @ (A @ C0).T
    return K


def _GGP_kernel_Nystrom(Q0:Tensor, A:Tensor, mask:Tensor) -> Tensor:
    Q = A @ Q0
    return Q

