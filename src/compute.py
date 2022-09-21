import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional
import numpy as np

def _sqrt_Nystrom(Q:Tensor, mask:Tensor) -> Tensor:
    """
    Compute the Nystrom approx square root of a kernel matrix.
    A smooth function is used for near-zero eigenvalues.
    The function is computed in-place.
    """
    D, V = torch.linalg.eigh(Q[mask])
    D_sqrt = torch.sqrt(torch.clamp(D, 0))
    D_sqrt_inv = D_sqrt / torch.clamp(D, D[-1] * 1e-4)
    Q[mask] = V * D_sqrt.view((1, -1))
    Q[~mask] = Q[~mask] @ V * D_sqrt_inv.view((1, -1))
    return Q


def _init_kernel(X:Tensor, kernel:str, **params) -> Tensor:
    if kernel == "linear":
        K0 = X @ X.T
    elif kernel == "rbf":
        K0 = torch.exp(-params["gamma"]*torch.cdist(X, X, p=2)**2)
    elif kernel == "laplacian":
        K0 = torch.exp(-params["gamma"]*torch.cdist(X, X, p=1))
    elif kernel == "arccos":
        K0 = 1 - torch.arccos(torch.corrcoef(X))/np.pi
    elif kernel == "sigmoid":
        K0 = torch.tanh(params["gamma"] * X @ X.T + params["c"])
    elif kernel == "polynomial":
        K0 = (X @ X.T + params["c"])**params["d"]
    else:
        raise Exception("Unsupported kernel function!")
    return K0


def _init_kernel_Nystrom(X:Tensor, mask:Tensor, kernel:str, **params) -> Tensor:
    if kernel == "linear":
        if torch.sum(mask) >= X.shape[1]: Q0 = X
        else: Q0 = _sqrt_Nystrom(X @ X[mask].T, mask)
    elif kernel == "rbf":
        K0 = torch.exp(-params["gamma"]*torch.cdist(X, X[mask], p=2)**2)
        Q0 = _sqrt_Nystrom(K0, mask)
    elif kernel == "laplacian":
        K0 = torch.exp(-params["gamma"]*torch.cdist(X, X[mask], p=1))
        Q0 = _sqrt_Nystrom(K0, mask)
    elif kernel == "arccos":
        X_scale = X - torch.mean(X, dim=0, keepdim=True)
        X_scale /= torch.sqrt(torch.sum(X_scale**2, dim=0, keepdim=True))
        K0 = torch.clamp(X_scale @ X_scale[mask].T, -1, 1)
        K0 = 1 - torch.arccos(K0)/np.pi
        Q0 = _sqrt_Nystrom(K0, mask)
    elif kernel == "sigmoid":
        K0 = torch.tanh(params["gamma"] * X @ X[mask].T + params["c"])
        Q0 = _sqrt_Nystrom(K0, mask)
    elif kernel == "polynomial":
        K0 = (X @ X[mask].T + params["c"])**params["d"]
        Q0 = _sqrt_Nystrom(K0, mask)
    else:
        raise Exception("Unsupported kernel function!")
    return Q0


def _add_layer(K:Tensor, A:Tensor, sigma_b:float, sigma_w:float, method:str, **params) -> Tensor:
    if method == "GCN":
        return _GCNConv(K, A, sigma_b, sigma_w)
    else:
        raise Exception("Unsupported layer type!")


def _add_layer_Nystrom(Q:Tensor, A:Tensor, sigma_b:float, sigma_w:float, mask:Tensor, method:str, **params) -> Tensor:
    if method == "GCN":
        return _GCNConv_Nystrom(Q, A, sigma_b, sigma_w, mask)
    else:
        raise Exception("Unsupported layer type!")


def _get_kernel(K0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, method:str, **params) -> Tensor:
    if method == "GCN":
        return _GCN_kernel(K0, A, L, sigma_b, sigma_w)
    elif method == "GCN2":
        return _GCN2_kernel(K0, A, L, sigma_b, sigma_w, **params)
    elif method == "GIN":
        return _GIN_kernel(K0, A, L, sigma_b, sigma_w, **params)
    elif method == "APPNP":
        return _APPNP_kernel(K0, A, L, sigma_b, sigma_w, **params)
    else:
        raise Exception("Unsupported layer type!")


def _get_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, method:str, **params) -> Tensor:
    if method == "GCN":
        return _GCN_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask)
    elif method == "GCN2":
        return _GCN2_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask, **params)
    elif method == "GIN":
        return _GIN_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask, **params)
    elif method == "APPNP":
        return _APPNP_kernel_Nystrom(Q0, A, L, sigma_b, sigma_w, mask, **params)
    else:
        raise Exception("Unsupported layer type!")


def _ExxT_ReLU(K:Tensor) -> Tensor:
    """
    Compute `_ExxT` with closed-form integral result for ReLU.
    """
    s = torch.sqrt(torch.diag(K)).view((-1, 1))
    theta = torch.arccos(torch.clamp(K/s/s.T, -1, 1))
    ExxT = 0.5/np.pi*(torch.sin(theta)+(np.pi-theta)*torch.cos(theta))*s*s.T
    return ExxT


def _ExxT_ReLU_Nystrom(Q:Tensor, mask:Tensor) -> Tensor:
    """
    Compute `_ExxT` with closed-form integral result for ReLU.
    Then, it will compute its Nystrom approx square root.
    """
    s = torch.sqrt(torch.sum(Q**2, 1)).view((-1, 1))
    theta = torch.arccos(torch.clamp(Q @ Q[mask].T/s/s[mask].T, -1, 1))
    ExxT = 0.5/np.pi*(torch.sin(theta)+(np.pi-theta)*torch.cos(theta))*s*s[mask].T
    return _sqrt_Nystrom(ExxT, mask)


def _GCNConv(K:Tensor, A:Tensor, sigma_b:float, sigma_w:float) -> Tensor:
    """
    Compute the kernel `K` of the next layer given its value at the current layer.

    Args:
        K (Tensor[float]): the `N * N` kernel matrix.
        A (Tensor[float]): the `N * N` adjacency matrix.
        sigma_b (float): bias variance of the current layer.
        sigma_w (float): weight variance of the current layer.
    """
    ExxT = _ExxT_ReLU(K)
    return sigma_b**2 + sigma_w**2 * A @ (A @ ExxT).T


def _GCNConv_Nystrom(Q:Tensor, A:Tensor, sigma_b:float, sigma_w:float, mask:Tensor) -> Tensor:
    """
    Compute the approx square root `Q` of the kernel `K` of the next layer given its value at the current layer.

    Args:
        Q (Tensor[float]): the `N * (Na+1)` square root of kernel matrix.
        A (Tensor[float]): the `N * N` adjacency matrix.
        sigma_b (float): bias variance of the current layer.
        sigma_w (float): weight variance of the current layer.
        mask (Tensor[bool]): the mask for landmark points.
    """
    ExxT = _ExxT_ReLU_Nystrom(Q, mask)
    return torch.cat((sigma_w * A @ ExxT, sigma_b * torch.ones_like(ExxT[:,[0]])), dim=1)


def _GCN_kernel(K0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float) -> Tensor:
    """
    Compute the GCN kernel.
    """
    K = sigma_b**2 + sigma_w**2 * A @ (A @ K0).T
    ExxT = torch.zeros_like(K)
    for j in range(L):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = sigma_b**2 + sigma_w**2 * A @ (A @ ExxT).T
    return K


def _GCN_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor) -> Tensor:
    """
    Compute the GCN kernel.
    """
    Na = torch.sum(mask)
    Q = torch.zeros((A.shape[0], Na+1), device=Q0.device)
    Q[:,:Q0.shape[1]] = sigma_w * A @ Q0 ; Q[:,-1] = sigma_b
    for j in range(L):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:-1] = sigma_w * A @ ExxT ; Q[:,-1] = sigma_b
    return Q


def _GCN2_kernel(K0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, alpha:float=0.1, beta:float=0.1) -> Tensor:
    """
    Compute the GCN2 kernel.
    """
    K = sigma_w**2 * A @ (A @ K0).T
    coef = (sigma_w*beta)**2+(1-beta)**2
    ExxT = torch.zeros_like(K)
    for j in range(L):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = coef * ((1-alpha)**2 * A @ (A @ (ExxT)).T + (alpha)**2 * K0)
    return K


def _GCN2_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, alpha:float=0.1, beta:float=0.1) -> Tensor:
    """
    Compute the result kernel by using `_GCNConv_ReLU` at each layer.
    """
    Na = torch.sum(mask); Ni = Q0.shape[1]
    Q = torch.zeros((A.shape[0], Na+Ni), device=Q0.device)
    Q[:,:Q0.shape[1]] = sigma_w * A @ Q0
    coef = np.sqrt((sigma_w*beta)**2+(1-beta)**2)
    for j in range(L):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:Na] = (1-alpha)*coef * A @ ExxT
        Q[:,Na:Na+Ni] = (alpha)*coef * Q0
    return Q


def _APPNP_kernel(K0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, alpha:float=0.1) -> Tensor:
    """
    Compute the result kernel by using `_GCNConv_ReLU` at each layer.
    """
    K = torch.zeros_like(K0)
    K[:] = K0
    for j in range(L):
        K[:] = A @ (A @ K).T * (1-alpha) + K0 * alpha
    ExxT = _ExxT_ReLU(K)
    return sigma_b**2 + sigma_w**2 * A @ (A @ ExxT).T


def _APPNP_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, alpha:float=0.1) -> Tensor:
    """
    Compute the result kernel by using `_GCNConv_ReLU` at each layer.
    """
    Na = torch.sum(mask); Ni = Q0.shape[1]
    Q = torch.zeros((A.shape[0], (L+1)*Ni), device=Q0.device)
    Q[:,:Ni] = Q0
    for j in range(L):
        Q[:] = A @ Q * np.sqrt(1-alpha)
        Q[:,(j+1)*Ni:(j+2)*Ni] = Q0 * np.sqrt(alpha)
    ExxT = _ExxT_ReLU_Nystrom(Q, mask)
    return torch.cat((sigma_w * A @ ExxT, sigma_b * torch.ones_like(ExxT[:,[0]])), dim=1)


def _GIN_kernel(K0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, eps:float=0.0) -> Tensor:
    """
    Compute the result kernel by using `_GCNConv_ReLU` at each layer.
    """
    K = sigma_b**2 + sigma_w**2 * A @ (A @ K0).T
    ExxT = torch.zeros_like(K)
    for j in range(L):
        ExxT[:] = _ExxT_ReLU(K)
        K[:] = sigma_b**2 + sigma_w**2 * ExxT
    return K


def _GIN_kernel_Nystrom(Q0:Tensor, A:Tensor, L:int, sigma_b:float, sigma_w:float, mask:Tensor, eps:float=0.0) -> Tensor:
    """
    Compute the result kernel by using `_GCNConv_ReLU` at each layer.
    """
    Na = torch.sum(mask)
    Q = torch.zeros((A.shape[0], Na+1), device=Q0.device)
    Q[:,:Q0.shape[1]] = sigma_w * A @ Q0 ; Q[:,-1] = sigma_b
    for j in range(L):
        ExxT = _ExxT_ReLU_Nystrom(Q, mask)
        Q[:,:-1] = sigma_w * ExxT ; Q[:,-1] = sigma_b
    return Q

