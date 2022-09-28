import torch
import torch.nn.functional as F
from tqdm import tqdm

from main_gnn import GCN, GCN2, GIN, SAGE, SGC

def main_batch(args, device, data, method):

    from torch_geometric.loader import NeighborLoader
    train_mask = data.train_mask.to('cpu')
    train_loader = NeighborLoader(data, input_nodes=train_mask, num_neighbors=[25, 10], shuffle=True, batch_size=16384, num_workers=16)
    subgraph_loader = NeighborLoader(data.clone(), input_nodes=None, num_neighbors=[-1], shuffle=False, batch_size=16384, num_workers=16)

    # No need to maintain these features during evaluation:
    del subgraph_loader.data.x, subgraph_loader.data.y
    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    def train():
        model.train()
        total_loss = total_correct = total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = model(batch)[:batch.batch_size]
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size
        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test():
        model.eval()
        y_hat = inference(model, data.x, subgraph_loader).argmax(dim=-1)
        y = data.y.cpu()
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            acc = torch.mean((y_hat[mask] == y[mask]).to(torch.float))
            accs.append(acc)
        return accs

    @torch.no_grad()
    def inference(model, x_all, subgraph_loader):
        only_edge_index = type(model).__name__ in ["GIN", "SAGE"]
        for i, conv in enumerate(model.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                if only_edge_index:
                    x = conv(x, batch.edge_index)
                else:
                    x = conv(x, batch.edge_index, batch.edge_weight)
                if i < len(model.convs) - 1: x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

    out_channels = data.num_classes
    if method == "GCN":
        model = GCN(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "GCN2":
        model = GCN2(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "GIN":
        model = GIN(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "SAGE":
        model = SAGE(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
    elif method == "SGC":
        model = SGC(data.num_features, args.dim_hidden, out_channels, args.num_layers, args.batchnorm, args.dropout).to(device)
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
            loss, approx_acc = train()
            train_acc, val_acc, test_acc = test()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            if args.verbose:
                print("Epoch: %03d, train: %.4f, val: %.4f, test: %.4f" % 
                      (epoch, train_acc, val_acc, test_acc))

        result_runs[j] = torch.tensor([process_time()-now, train_acc, best_val_acc, best_test_acc])
        now = process_time()
        print("Run: %02d," % (j+1), result_format(result_runs[j]))
        model.reset_parameters()

    if args.runs > 1:
        print('----')
        print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
        print("SD:     ", result_format(torch.std(result_runs, dim=0)))
    print('----')

