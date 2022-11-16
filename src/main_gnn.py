import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, BatchNorm1d
from torch_geometric.nn import GCNConv, GCN2Conv, GINConv, SAGEConv

class GCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, batchnorm, dropout):
        super(GCN, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.bns = ModuleList()
        self.bns.append(BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.batchnorm: x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class GCN2(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, batchnorm, dropout):
        super(GCN2, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        from torch.nn import Linear
        self.convs = ModuleList()
        self.convs.append(Linear(in_channels, hidden_channels))
        self.bns = ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=i+1, normalize=False))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.convs.append(Linear(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](x)
        x = x_0 = F.relu(x)
        for i, conv in enumerate(self.convs[1:-1]):
            x = conv(x, x_0, edge_index, edge_weight)
            if self.batchnorm: x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class GIN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, batchnorm, dropout):
        super(GIN, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
        self.nn = Sequential()
        self.nn.append(Linear(in_channels, hidden_channels))
        if self.dropout: self.nn.append(BatchNorm1d(hidden_channels))
        self.nn.append(ReLU())
        self.nn.append(Dropout(p=self.dropout))
        for _ in range(num_layers-2):
            self.nn.append(Linear(hidden_channels, hidden_channels))
            if self.dropout: self.nn.append(BatchNorm1d(hidden_channels))
            self.nn.append(ReLU())
            self.nn.append(Dropout(p=self.dropout))
        self.nn.append(Linear(hidden_channels, out_channels))
        self.convs = ModuleList()
        self.convs.append(GINConv(nn=self.nn))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class SAGE(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, batchnorm, dropout):
        super(SAGE, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = ModuleList()
        self.bns.append(BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


def main_gnn(args, device, data, method):

    def nll_loss(model, data):
        preds = model(data).log_softmax(dim=-1)
        return F.nll_loss(preds[data.train_mask], data.y[data.train_mask])
    
    def accuracy(preds, y):
        return torch.mean((preds.argmax(dim=-1) == y).to(torch.float))
    
    def mse_loss(model, data):
        preds = model(data)
        return F.mse_loss(preds[data.train_mask,0], data.y[data.train_mask])

    def rsquared(preds, y):
        return 1 - torch.sum((preds[:,0]-y)**2) / torch.sum((y-torch.mean(y))**2)

    if data.y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        out_channels = data.num_classes
        train_loss = nll_loss
        test_metric = accuracy
    else:
        out_channels = 1
        train_loss = mse_loss
        test_metric = rsquared

    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        loss = train_loss(model, data)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(model, data):
        model.eval()
        preds = model(data)
        result = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            acc = test_metric(preds[mask], data.y[mask])
            result.append(acc)
        return result

    if method == "GCN":
        model = GCN(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "GCN2":
        model = GCN2(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "GIN":
        model = GIN(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "SAGE":
        model = SAGE(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    else:
        raise Exception("Unsupported GNN architecture!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.manual_seed(123)

    def result_format(result):
        return "time: %.4f, train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2], result[3])

    from time import process_time
    now = process_time()
    result_runs = torch.zeros((args.runs, 4))

    for j in range(args.runs):
        best_val_acc = best_test_acc = 0
        for epoch in range(1, 1+args.epochs):
            loss = train(model, data, optimizer)
            train_acc, val_acc, test_acc = test(model, data)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            if epoch % 10 == 0 and args.verbose:
                print("Epoch: %03d, train: %.4f, val: %.4f, test: %.4f" % 
                      (epoch, train_acc, val_acc, test_acc))

        result_runs[j] = torch.tensor([process_time()-now, train_acc, best_val_acc, best_test_acc])
        now = process_time()
        print("Run: %02d," % (j+1), result_format(result_runs[j]))
        model.reset_parameters()

    if args.runs > 1:
        print("----")
        print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
        print("SD:     ", result_format(torch.std(result_runs, dim=0)))
    print("----")

