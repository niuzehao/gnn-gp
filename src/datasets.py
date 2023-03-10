import os.path as osp

import torch
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T

from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data

class transition_matrix(BaseTransform):
    """
    Returns a transform that applies normalization on a given sparse matrix.

    Args:
        self_loop_weight (float): weight of the added self-loop. (default: `1`)
        normalization (str): Normalization scheme. supported values:
            "sym": symmetric normalization $\hat{A}=D^{-1/2}AD^{-1/2}$.
            "col": column-wise normalization $\hat{A}=AD^{-1}$.
            "row": row-wise normalization $\hat{A}=D^{-1}A$.
            others: No normalization.
            (default: "sym")
    """
    def __init__(self, self_loop_weight=1, normalization="sym"):
        self.self_loop_weight = self_loop_weight
        self.normalization = normalization
    
    def __call__(self, data):
        N = data.num_nodes
        A = data.edge_index
        if data.edge_attr is None:
            edge_weight = torch.ones(A.size(1), device=A.device)
        else:
            edge_weight = data.edge_attr

        from torch_geometric.utils import add_self_loops
        if self.self_loop_weight:
            A, edge_weight = add_self_loops(
                A, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        A, edge_weight = coalesce(A, edge_weight, N, N)

        from torch_scatter import scatter_add
        if self.normalization == "sym":
            row, col = A
            deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif self.normalization == "col":
            _, col = A
            deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float("inf")] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif self.normalization == "row":
            row, _ = A
            deg = scatter_add(edge_weight, row, dim=0, dim_size=N)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float("inf")] = 0
            edge_weight = edge_weight * deg_inv[row]
        else:
            pass

        data.edge_index = A
        data.edge_attr = edge_weight

        return data


class Scale(BaseTransform):
    def __init__(self, center=True, scale=False):
        self.center = center
        self.scale = scale

    def __call__(self, data):
        if self.center:
            data.x -= torch.mean(data.x, dim=0, keepdim=True)
        if self.scale:
            data.x /= torch.sqrt(torch.sum(data.x**2, dim=0, keepdim=True))
        return data


class WikipediaNetwork(InMemoryDataset):
    """
    The Wikipedia networks used in the
    `"Multi-Scale Attributed Node Embedding"<https://github.com/benedekrozemberczki/MUSAE>`.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The target is the logrithm of the number of average monthly traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset, supported values:
            "Chameleon", "Squirrel", or "Crocodile".
        transform: A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: `None`)
        pre_transform: A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: `None`)
    """

    raw_url = "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input"
    split_url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits"

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ["chameleon", "crocodile", "squirrel"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        raw_file_names = [f"{self.name}.json", f"{self.name}_edges.csv", f"{self.name}_target.csv"]
        if self.name != "crocodile":
            raw_file_names += [f"{self.name}_split_0.6_0.2_{i}.npz" for i in range(10)]
        return raw_file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        download_url(f"{self.raw_url}/features/{self.name}.json", self.raw_dir)
        download_url(f"{self.raw_url}/edges/{self.name}_edges.csv", self.raw_dir)
        download_url(f"{self.raw_url}/target/{self.name}_target.csv", self.raw_dir)
        for filename in self.raw_file_names[2:]:
            download_url(f"{self.split_url}/{filename}", self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], "r") as f:
            data = eval(f.read())
            values = sorted(set().union(*data.values()))
            x = [[v in data[str(r)] for v in values] for r in range(len(data))]
            x = torch.tensor(x, dtype=torch.float)
        with open(self.raw_paths[1], "r") as f:
            # A symmetrization is required for the graph.
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split(",")] for r in data]
            values = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = torch.unique(torch.cat((values, values.flip(0)), dim=1), dim=1)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        with open(self.raw_paths[2], "r") as f:
            data = f.read().split("\n")[1:-1]
            y = [int(r.split(",")[1]) for r in data]
            y = torch.log(torch.tensor(y))
        
        if self.name != "crocodile":
            from numpy import load
            f = load(self.raw_paths[3])
            train_mask = torch.from_numpy(f["train_mask"]).to(torch.bool)
            val_mask = torch.from_numpy(f["val_mask"]).to(torch.bool)
            test_mask = torch.from_numpy(f["test_mask"]).to(torch.bool)
        else:
            torch.manual_seed(123)
            mask = torch.randint_like(y, 25)
            train_mask = mask < 12
            val_mask = (mask >= 12) & (mask < 20)
            test_mask = mask >= 20
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class OGB_arxiv(InMemoryDataset):
    """
    The ogbn dataset from the `"Open Graph Benchmark: Datasets for
    Machine Learning on Graphs" <https://arxiv.org/abs/2005.00687>` paper.
    ogbn-arxiv is a paper citation network of arXiv papers.
    Each node is an ArXiv paper and each directed edge indicates that one paper cites another one.
    Node features are 128-dimensional vector obtained by averaging the WORD2VEC embeddings of words in its title and abstract.
    The task is to predict the 40 subject areas of ARXIV CS papers.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: `None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: `None`)
    """

    url = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"

    def __init__(self, root: str, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return int(self.data.y.max()) + 1

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "arxiv", "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "arxiv", "processed")

    @property
    def raw_file_names(self) -> str:
        file_names = ["node-feat.csv.gz", "node-label.csv.gz", "edge.csv.gz", "train.csv.gz", "valid.csv.gz", "test.csv.gz"]
        return file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        import os, shutil
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        for file_name in self.raw_file_names[:3]:
            path = osp.join(self.raw_dir, "arxiv", "raw", file_name)
            shutil.move(path, self.raw_dir)
        for file_name in self.raw_file_names[3:]:
            path = osp.join(self.raw_dir, "arxiv", "split", "time", file_name)
            shutil.move(path, self.raw_dir)
        shutil.rmtree(osp.join(self.raw_dir, "arxiv"))
        os.remove(osp.join(self.raw_dir, "arxiv.zip"))

    def process(self):
        import pandas as pd
        import numpy as np

        values = pd.read_csv(self.raw_paths[0], compression="gzip", header=None, dtype=np.float32).values
        x = torch.from_numpy(values)

        values = pd.read_csv(self.raw_paths[1], compression="gzip", header=None, dtype=np.int64).values
        y = torch.from_numpy(values).view(-1)

        # A symmetrization is required for the graph.
        values = pd.read_csv(self.raw_paths[2], compression="gzip", header=None, dtype=np.int64).values
        values = torch.from_numpy(values).t().contiguous()
        edge_index = torch.unique(torch.cat((values, values.flip(0)), dim=1), dim=1)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        for f, v in [("train", "train"), ("valid", "val"), ("test", "test")]:
            values = pd.read_csv(f"{self.raw_dir}/{f}.csv.gz", compression="gzip", header=None, dtype=np.int64).values
            idx = torch.from_numpy(values).view(-1)
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f"{v}_mask"] = mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def load_data(name, path=osp.join(osp.abspath(__file__), "..", "..", "data"),
              center=False, scale=False, transform=None):
    """
    Load dataset and save to disk. Automatic downloads may occur upon first use.

    Args:
        name (str): name for the dataset. supported values:
            "Cora", "CiteSeer", "PubMed": Planetoid dataset.
            "chameleon", "crocodile", "squirrel": Wikipedia dataset.
            "arxiv": OGBN dataset.
            "Reddit": reddit dataset.
        path (str): directory where the dataset should be saved.
        center (bool): center each column to have mean 0.
        scale (bool): scale each column to have standard deviation 1.
        transform (callable, optional): a function/transform to a graph data object.
            The data object will be transformed after centering and scaling.
    """
    if name in ["Cora", "CiteSeer", "PubMed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name)
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif name in ["chameleon", "crocodile", "squirrel"]:
        dataset = WikipediaNetwork(path, name)
        data = dataset[0]
    elif name == "arxiv":
        dataset = OGB_arxiv(path)
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif name == "Reddit":
        from torch_geometric.datasets.reddit import Reddit
        datasets = Reddit(osp.join(path,"Reddit"))
        data = datasets[0]
        data.num_classes = datasets.num_classes
    else: raise Exception("Unsupported Dataset!")
    if center or scale: norm = Scale(center, scale); data = norm(data)
    if transform is not None: data = transform(data)
    return data


def get_data(data):
    """
    Return attributes from a homogeneous graph.

    Args:
        data (torch_geometric.data.data): a graph data object.
    """
    X = data.x
    y = data.y
    N = X.shape[0]
    A = data.edge_index
    if data.edge_attr is None:
        edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    else:
        edge_weight = data.edge_attr
    A = torch.sparse_coo_tensor(data.edge_index, edge_weight, (N, N))
    mask = {"train":data.train_mask, "val":data.val_mask, "test":data.test_mask}
    return N, X, y, A, mask

