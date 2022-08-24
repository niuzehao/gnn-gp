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
    eye = torch.mean(torch.diag(K))*torch.eye(torch.sum(mb), device=K.device)
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
    eye = torch.mean(torch.sum(Q**2, 1))*torch.eye(torch.sum(ma), device=Q.device)
    K = Q @ Q[ma].T
    temp1 = torch.linalg.lstsq(K[ma], K[mb].T, rcond=1e-4).solution
    temp2 = temp1 @ K[mb]
    # for i, eps in enumerate(nugget):
    #     fitted[i] = K @ torch.linalg.solve(temp2 + eps*eye, temp1 @ yb)
    w, v = torch.linalg.eigh(temp2)
    d = torch.mean(torch.sum(Q**2, 1))
    temp3 = K @ v
    temp4 = v.T @ temp1 @ yb
    for i, eps in enumerate(nugget):
        fitted[i] = temp1 @ torch.diag(1/(w+eps*d)) @ temp2
    return fitted[0] if len(nugget) == 1 else fitted


def error(fit:Tensor, y:Tensor, mask:Dict[str, Tensor], loss:Union[str,List[str]]):
    """
    Compute the average loss for the dataset.

    Args:
        fit (Tensor[float]): predicted value of the dataset.
        y (Tensor[float] or Tensor[int]): prediction target.
            For float type, a regression result will be generated.
            For int type, a classification result will be generated.
        mask (Dict[str, Tensor[bool]]): the training, validation and test mask.
        loss (str or List[str]): loss metric. supported values:
            "mse": mean squared error
            "mae": mean absolute error
            "mr": misclassification rate
            "nll": negative log likelihood
    """
    if isinstance(loss, str): loss = [loss]
    func = {}
    if 'mse' in loss: func['mse'] = lambda fit, y: (fit-y)**2
    if 'mae' in loss: func['mae'] = lambda fit, y: torch.abs(fit-y)
    if 'mr' in loss:  func['mr']  = lambda fit, y: (torch.argmax(fit, 1) != y).to(torch.float)
    if 'nll' in loss: func['nll'] = lambda fit, y: -torch.log(fit[torch.arange(len(y)),y])
    num_nugget = fit.shape[0]
    err = {key:{} for key in func}
    for key in func:
        for subset in mask:
            err[key][subset] = torch.zeros(num_nugget)
        for i in range(num_nugget):
            y_err = func[key](fit[i], y)
            for subset in mask:
                err[key][subset][i] = torch.mean(y_err[mask[subset]])
    return err

