import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional


def fit(K:Tensor, y:Tensor, train_mask:Tensor, nugget:Union[float,List[float]]=1e-2):
    """
    Making predictions using training data, for each given nugget.
    For classification problems, the prediction target is classification probability.
    For regression problems, the prediction target is output value.

    Args:
        K (Tensor[float]): the GNNGP Kernel matrix.
        y (Tensor[float] or Tensor[int]): prediction target.
            For int type, the task is a classification problem.
            For float type, the task is a regression problem.
        train_mask (Tensor[bool]): mask for training points.
        nugget (float or Tensor[float]): the nugget used in posterior inference.
            (default: 1e-2 x mean diagonal element of K)
    """
    mb = train_mask
    if isinstance(nugget, float): nugget = [nugget]
    if y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        yb = torch.nn.functional.one_hot(y[mb]).to(torch.float)
    else: yb = y[mb]
    fitted = torch.zeros((len(nugget), K.shape[0], *yb.shape[1:]), device=K.device)
    w, v = torch.linalg.eigh(K[mb][:,mb])
    d = torch.mean(torch.diag(K))
    shape = (-1, 1) if len(yb.shape) >= 2 else (-1,)
    temp = v.T @ yb
    for i, eps in enumerate(nugget):
        fitted[i] = K[:,mb] @ (v @ (temp/(w+eps*d).view(shape)))
    return fitted[0] if len(nugget) == 1 else fitted


def fit_Nystrom(Q:Tensor, y:Tensor, train_mask:Tensor, landmark_mask:Tensor, nugget:Union[float,List[float]]=1e-2):
    """
    Making predictions using training data, for each given nugget.
    For classification problems, the prediction target is classification probability.
    For regression problems, the prediction target is output value.

    Args:
        Q (Tensor[float]): square root of the GNNGP Kernel matrix.
        y (Tensor[float] or Tensor[int]): prediction target.
            For int type, the task is a classification problem.
            For float type, the task is a regression problem.
        train_mask (Tensor[bool]): mask for training points.
        landmark_mask (Tensor[bool]): mask for landmark points.
        nugget (float or Tensor[float]): the nugget used in posterior inference.
            (default: 1e-2 x mean diagonal element of QQ^T)
    """
    ma = landmark_mask; mb = train_mask
    if isinstance(nugget, float): nugget = [nugget]
    if y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        yb = torch.nn.functional.one_hot(y[mb]).to(torch.float)
    else: yb = y[mb]
    fitted = torch.zeros((len(nugget), Q.shape[0], *yb.shape[1:]), device=Q.device)
    w, v = torch.linalg.eigh(Q[mb].T @ Q[mb])
    d = torch.mean(torch.sum(Q**2, 1))
    shape = (-1, 1) if len(yb.shape) >= 2 else (-1,)
    temp = v.T @ (Q[mb].T @ yb)
    for i, eps in enumerate(nugget):
        fitted[i] = Q @ (v @ (temp/(w+eps*d).view(shape)))
    return fitted[0] if len(nugget) == 1 else fitted


def result(fit:Tensor, y:Tensor, masks:Dict[str, Tensor]):
    """
    Compute the train, validation and test result for each given nugget.
    For classification problems, the metric is mean classification accuracy.
    For regression problems, the metric is the R-squared statistic.

    Args:
        fit (Tensor[float]): model predicted value.
        y (Tensor[float] or Tensor[int]): prediction target.
            For int type, the task is a classification problem.
            For float type, the task is a regression problem.
        masks (Dict[str, Tensor[bool]]): the training, validation, test and possibly landmark masks.
    """
    if y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        func = lambda fit, y: torch.mean((torch.argmax(fit, 1) == y).to(torch.float))
    else:
        func = lambda fit, y: 1 - torch.sum((fit-y)**2) / torch.sum((y-torch.mean(y))**2)
    num_nugget = fit.shape[0]
    result = {}
    for subset in masks:
        result[subset] = torch.zeros(num_nugget)
    for i in range(num_nugget):
        loss = func(fit[i], y)
        for subset, mask in masks.items():
            result[subset][i] = func(fit[i][mask], y[mask])
    return result

