import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

from argparse import ArgumentParser
from time import process_time

def main():
    parser = ArgumentParser(description='GNNGP arguments')
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--action', required=True, choices=['gnn', 'gp', 'rbf'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runs', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')

    # preprocessing
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--scale', action='store_true')

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
    methods = ['GCN', 'GCN2', 'GIN', 'SAGE', 'SGC']

    for name in dnames:
        if name.lower().startswith(args.data.lower()):
            print("Dataset: %s" % name)
            break
    else:
        raise Exception("Unsupported data! Possible values:", ' '.join(dnames))

    if args.action != 'rbf':
        for method in methods:
            if method.lower() == args.method.lower():
                print("Method: %s%s" % (method, '-GP' if args.action == 'gp' else ''))
                break
        else:
            raise Exception("Unsupported method! Possible values:", ' '.join(method))
    else:
        print("Method: RBF")

    transform = transition_matrix() if args.action == 'gp' else None

    device = torch.device('cuda:%s' % args.device if args.device>=0 else 'cpu')
    data = load_data(name, center=args.center, scale=args.scale, transform=transform)
    data = data.to(device)
    print("Dataset loaded to %s" % device)

    torch.manual_seed(123)
    if args.fraction > 0:
        print("Random select %.0f percent of landmark points" % (100/args.fraction))
    if args.runs <= 0:
        if args.action != 'gnn' and args.fraction <= 1:
            print("No randomness. Run only once.")
            args.runs = 1
        else:
            print("Number of runs not given. Set to 10 runs.")
            args.runs = 10
    print('----')
    
    def result_format(result):
        return "time: %.4f, train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2], result[3])
    
    if args.action == 'gp':
        now = process_time()
        result_runs = torch.zeros((args.runs, 4))
        epsilon = torch.logspace(-3, 1, 101, device=device)

        model = GNNGP(data, args.L, args.sigma_b, args.sigma_w, device=args.device, Nystrom=args.fraction > 0)

        for j in range(args.runs):
            if args.fraction > 0:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            model.computed = False
            model.get_kernel(method=method)
            result = model.get_result(epsilon)
            i = torch.argmax(result["val"])
            result_runs[j] = torch.tensor([process_time()-now, result["train"][i], result["val"][i], result["test"][i]])
            now = process_time()
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print('----')
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print('----')

    if args.action == 'rbf':
        from compute import _sqrt_Nystrom
        now = process_time()
        result_runs = torch.zeros((args.runs, 4))
        epsilon = torch.logspace(-3, 1, 101, device=device)
        model = GNNGP(data, 0, 0.0, 1.0, device=args.device, Nystrom=args.fraction > 0)
        model.computed = True
        if args.fraction <= 0:
            dist = torch.cdist(model.X, model.X, p=2)
        elif args.fraction == 1:
            model.mask["landmark"] = model.mask["train"]
            mask = model.mask["landmark"]
            dist = torch.cdist(model.X, model.X[mask], p=2)
        for j in range(args.runs):
            if args.fraction > 1:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
                mask = model.mask["landmark"]
                dist = torch.cdist(model.X, model.X[mask], p=2)
            for gamma in torch.logspace(-2, 2, 101):
                K0 = torch.exp(-gamma*dist**2)
                if args.fraction > 0:
                    model.Q = _sqrt_Nystrom(K0, mask)
                else:
                    model.K = K0
                result = model.get_result(epsilon)
                i = torch.argmax(result["val"])
                if result["val"][i] > result_runs[j][1]:
                    result_runs[j] = torch.tensor([0.0, result["train"][i], result["val"][i], result["test"][i]])
            result_runs[j][0] = process_time()-now
            now = process_time()
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print('----')
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print('----')


    if args.action == 'gnn':
        from main_gnn import main_gnn
        main_gnn(args, device, data, method)

if __name__ == "__main__":
    main()
