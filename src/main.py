import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

from argparse import ArgumentParser
from time import process_time

def main():
    parser = ArgumentParser(description="GNNGP arguments")
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--action", required=True, choices=["gnn", "gp", "rbf"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # preprocessing
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--scale", action="store_true")

    # GP arguments
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--sigma_b", type=float, default=0.0)
    parser.add_argument("--sigma_w", type=float, default=1.0)
    parser.add_argument("--fraction", type=int, default=0)

    # GNN arguments
    parser.add_argument("--dim_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=0)

    args = parser.parse_args()
    print(args)

    dnames = ["Cora", "CiteSeer", "PubMed", "chameleon", "crocodile", "squirrel", "arxiv", "Reddit"]
    methods = ["GCN", "GCN2", "GIN", "SAGE", "GGP"]

    for name in dnames:
        if name.lower().startswith(args.data.lower()):
            print("Dataset: %s" % name)
            break
    else:
        raise Exception("Unsupported data! Possible values: " + " ".join(dnames))

    if args.action in ["gnn", "gp"]:
        for method in methods:
            if method.lower() == args.method.lower():
                print("Method: %s%s" % (method, "-GP" if args.action == "gp" else ""))
                break
        else:
            raise Exception("Unsupported method! Possible values: " + " ".join(methods))
    else:
        print("Method: RBF")

    transform = transition_matrix(normalization_in="sym" if method not in ["SAGE", "GGP"] else "row")
    data = load_data(name, center=args.center, scale=args.scale, transform=transform)
    device = torch.device("cuda:%s" % args.device if args.device>=0 else "cpu")
    data = data.to(device)
    print("Dataset loaded to %s" % device)

    torch.manual_seed(123)
    if args.fraction > 0:
        print("Random select %.0f percent of landmark points" % (100/args.fraction))
    if args.runs <= 0:
        if args.action != "gnn" and args.fraction <= 1:
            print("No randomness. Run only once.")
            args.runs = 1
        else:
            print("Number of runs not given. Set to 10 runs.")
            args.runs = 10
    print("----")
    
    def result_format(result):
        return "time: %.4f, train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2], result[3])
    
    if args.action == "gp":
        now = process_time()
        result_runs = torch.zeros((args.runs, 4))
        epsilon = torch.logspace(-3, 1, 101, device=device)
        params = {} if method != "GGP" else {"kernel":"polynomial", "c":5.0, "d":3.0}
        L = args.num_layers - 1 # number of hidden layers
        model = GNNGP(data, L, args.sigma_b, args.sigma_w, device=args.device, Nystrom=args.fraction > 0)
        for j in range(args.runs):
            if args.fraction > 0:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            model.computed = False
            model.get_kernel(method=method, **params)
            model.predict(epsilon)
            summary = model.get_summary()
            result_runs[j] = torch.tensor([process_time()-now, summary["train"], summary["val"], summary["test"]])
            now = process_time()
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print("----")
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print("----")

    if args.action == "rbf":
        now = process_time()
        result_runs = torch.zeros((args.runs, 4))
        epsilon = torch.logspace(-3, 1, 101, device=device)
        model = GNNGP(data, 0, 0.0, 1.0, device=args.device, Nystrom=args.fraction > 0)
        for j in range(args.runs):
            if args.fraction > 0:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            for gamma in torch.logspace(-2, 2, 101):
                model.set_init_kernel(kernel="rbf", gamma=gamma)
                if args.fraction > 0:
                    model.Q = model.Q0
                else:
                    model.K = model.K0
                model.computed = True
                model.predict(epsilon)
                summary = model.get_summary()
                if summary["val"] > result_runs[j][2]:
                    result_runs[j] = torch.tensor([0.0, summary["train"], summary["val"], summary["test"]])
            result_runs[j][0] = process_time()-now
            now = process_time()
            print("Run: %02d," % (j+1), result_format(result_runs[j]))
        if args.runs > 1:
            print("----")
            print("Mean:   ", result_format(torch.mean(result_runs, dim=0)))
            print("SD:     ", result_format(torch.std(result_runs, dim=0)))
        print("----")

    if args.action == "gnn":
        args.batchnorm = name == "arxiv"
        if args.batch_size <= 0:
            from main_gnn import main_gnn
            main_gnn(args, device, data, method)
        else:
            from main_batch import main_batch
            main_batch(args, device, data, method)

if __name__ == "__main__":
    main()
