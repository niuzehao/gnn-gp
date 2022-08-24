import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

def gnn_mr(args, device, data, method, verbose=False):
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                     method="GCN"):
            super(GNN, self).__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        def forward(self, data):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index, edge_weight)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            x = self.convs[-1](x, edge_index, edge_weight)
            return x.log_softmax(dim=-1)

        def reset_parameters(self):
            for conv in self.convs:
                conv.reset_parameters()

    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(model, data):
        model.eval()
        preds = model(data)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = preds[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    data = data.to(device)
    model = GNN(data.num_features, args.dim_hidden,
                data.num_classes, args.num_layers,
                method=args.method.upper()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.manual_seed(123)

    import time
    now = time.process_time()
    result_runs = torch.zeros((args.runs, 3))

    def result_format(result):
        return "train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2])

    for j in range(args.runs):
        best_val_acc = test_acc = 0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, optimizer)
            result = test(model, data)

            train_acc, val_acc, tmp_test_acc = result
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            if epoch % 10 == 0 and verbose:
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, 1-train_acc, 1-best_val_acc, 1-test_acc))

        result_runs[j] = torch.tensor([1-train_acc, 1-best_val_acc, 1-test_acc])
        print("Run: %02d," % (j+1), result_format(result_runs[j]))
        model.reset_parameters()

    if args.runs > 1:
        print('---')
        print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
        print("SD:     ", result_format(torch.std(result_runs, dim=0)))
    print('---')
    print("Time spent per run: %.4f" % ((time.process_time()-now)/args.runs))

