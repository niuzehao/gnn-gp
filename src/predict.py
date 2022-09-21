import torch
from torch import Tensor
from typing import Callable, Dict, List, Union, Optional


def fit(K:Tensor, y:Tensor, train_mask:Tensor, nugget:Union[float,List[float]]=1e-2):
    """
    Making predictions of target based on training data.
    For classification problems, the one-hot encoding is applied,
    and the prediction target will be the classification probabilities.

    Args:
        K (Tensor[float]): kernel matrix of the dataset.
        y (Tensor[float] or Tensor[int]): prediction target.
            For float type, a regression result will be generated.
            For int type, a classification result will be generated.
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
    # eye = torch.mean(torch.diag(K))*torch.eye(torch.sum(mb), device=K.device)
    # for i, eps in enumerate(nugget):
    #     fitted[i] = K[:,mb] @ torch.linalg.solve(K[mb][:,mb] + eps*eye, yb)
    w, v = torch.linalg.eigh(K[mb][:,mb])
    d = torch.mean(torch.diag(K))
    temp1 = K[:,mb] @ v
    temp2 = v.T @ yb
    for i, eps in enumerate(nugget):
        fitted[i] = temp1 @ torch.diag(1/(w+eps*d)) @ temp2
    return fitted[0] if len(nugget) == 1 else fitted


def fit_Nystrom(Q:Tensor, y:Tensor, train_mask:Tensor, mask:Tensor, nugget:Union[float,List[float]]=1e-2):
    """
    Making predictions of target based on training data.
    For classification problems, the one-hot encoding is applied,
    and the prediction target will be the classification probabilities.

    Args:
        Q (Tensor[float]): square root of kernel matrix.
        y (Tensor[float] or Tensor[int]): prediction target.
            For float type, a regression result will be generated.
            For int type, a classification result will be generated.
        train_mask (Tensor[bool]): mask for training points.
        mask (Tensor[bool]): mask for landmark points.
        nugget (float or Tensor[float]): the nugget used in posterior inference.
            (default: 1e-2 x mean diagonal element of QQ^T)
    """
    ma = mask; mb = train_mask
    if isinstance(nugget, float): nugget = [nugget]
    if y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        yb = torch.nn.functional.one_hot(y[mb]).to(torch.float)
    else: yb = y[mb]
    fitted = torch.zeros((len(nugget), Q.shape[0], *yb.shape[1:]), device=Q.device)
    w, v = torch.linalg.eigh(Q[mb].T @ Q[mb])
    d = torch.mean(torch.sum(Q**2, 1))
    temp1 = Q @ v
    temp2 = v.T @ (Q[mb].T @ yb)
    for i, eps in enumerate(nugget):
        fitted[i] = temp1 @ torch.diag(1/(w+eps*d)) @ temp2
    return fitted[0] if len(nugget) == 1 else fitted


def error(fit:Tensor, y:Tensor, mask:Dict[str, Tensor]):
    """
    Compute the average loss for the dataset.
    For classification problems, the loss function is misclassification rate.
    For regression problems, the loss function is the mean square error.

    Args:
        fit (Tensor[float]): predicted value of the dataset.
        y (Tensor[float] or Tensor[int]): prediction target.
            For float type, a regression result will be generated.
            For int type, a classification result will be generated.
        mask (Dict[str, Tensor[bool]]): the training, validation and test mask.
    """
    if y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        func = lambda fit, y: (torch.argmax(fit, 1) != y).to(torch.float)
    else:
        func = lambda fit, y: (fit-y)**2
    num_nugget = fit.shape[0]
    error = {}
    for subset in mask:
        error[subset] = torch.zeros(num_nugget)
    for i in range(num_nugget):
        loss = func(fit[i], y)
        for subset in mask:
            error[subset][i] = torch.mean(loss[mask[subset]])
    return error

