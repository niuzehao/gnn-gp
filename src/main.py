import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

import argparse
import time

def result_format(result):
    return "train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2])


def main():
    parser = argparse.ArgumentParser(description='GNNGP arguments')
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--action', required=True, choices=['gnn', 'gp', 'rbf'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)

    # preprocessing
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--train_size', type=int, default=0)

    # GP arguments
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--sigma_b', type=float, default=0.0)
    parser.add_argument('--sigma_w', type=float, default=1.0)
    parser.add_argument('--fraction', type=int, default=0)

    # GNN arguments
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    print(args)

    dnames = ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'crocodile', 'squirrel', 'arxiv', 'Reddit']
    methods = ['GCN', 'GCN2', 'GIN', 'SAGE', 'SGConv']

    for name in dnames:
        if name.lower().startswith(args.data.lower()):
            print("Dataset: %s" % name)
            break

    for method in methods:
        if method.lower() == args.method.lower():
            print("Method: %s" % method)
            break

    transform = transition_matrix() if args.action == 'gp' else None

    device = torch.device('cuda:%s' % args.device if args.device>=0 else 'cpu')
    data = load_data(name, center=args.center, scale=args.scale, transform=transform)
    data = data.to(device)
    print("Dataset loaded to %s" % device)

    torch.manual_seed(123)
    if args.train_size > 0:
        print("Random select %.0f percent of training points" % (100*args.train_size/torch.sum(data.train_mask)))
    if args.fraction > 0:
        print("Random select %.0f percent of landmark points" % (100/args.fraction))
    if args.action != 'gnn' and args.train_size <= 0 and args.fraction <= 0:
        print("No randomness. Run only once.")
        args.runs = 1
    
    if args.action == 'gp':
        torch.manual_seed(123)
        now = time.process_time()
        result_runs = torch.zeros((args.runs, 3))
        epsilon = torch.logspace(-3, 1, 101, device=device)

        model = GNNGP(data, args.L, args.sigma_b, args.sigma_w, device=args.device, Nystrom=args.fraction > 0)

        for j in range(args.runs):
            if args.train_size > 0:
                model.mask["train"] = data.train_mask & (torch.rand(model.N, device=device) < args.train_size/torch.sum(data.train_mask))
            if args.fraction > 0:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            model.computed = False
            model.get_kernel(method=method)
            result = model.get_error(epsilon)
            i = torch.argmin(result["val"])
            result_runs[j] = torch.tensor([result["train"][i], result["val"][i], result["test"][i]])
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print('----')
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print('----')
        print("Time spent per run: %.4f" % ((time.process_time()-now)/args.runs))

    if args.action == 'rbf':
        from compute import _sqrt_Nystrom
        torch.manual_seed(123)
        now = time.process_time()
        result_runs = torch.full((args.runs, 3), torch.inf)
        epsilon = torch.logspace(-3, 1, 101, device=device)
        model = GNNGP(data, 0, 0.0, 1.0, device=args.device, Nystrom=args.fraction > 0)
        model.computed = True
        for j in range(args.runs):
            if args.fraction > 0:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
                mask = model.mask["landmark"]
                dist = torch.cdist(model.X, model.X[mask], p=2)
            else:
                dist = torch.cdist(model.X, model.X, p=2)
            model.set_init_kernel()
            for gamma in torch.logspace(-2, 2, 101):
                K0 = torch.exp(-gamma*dist**2)
                if args.fraction > 0:
                    model.Q = _sqrt_Nystrom(K0, mask)
                else:
                    model.K = model.K0
                result = model.get_error(epsilon)
                i = torch.argmin(result["val"])
                if result["val"][i] < result_runs[j][1]:
                    result_runs[j] = torch.tensor([result["train"][i], result["val"][i], result["test"][i]])
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print('----')
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print('----')
        print("Time spent per run: %.4f" % ((time.process_time()-now)/args.runs))


    if args.action == 'gnn':
        from main_gnn import gnn_mr
        gnn_mr(args, device, data, method)

if __name__ == "__main__":
    main()
