import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="GNNGP arguments")
    parser.add_argument("data")
    parser.add_argument("method")
    parser.add_argument("action")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # feature preprocessing
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
    args.Nystrom = args.fraction > 0
    args.action = args.action.upper()

    dnames = ["Cora", "CiteSeer", "PubMed", "chameleon", "crocodile", "squirrel", "arxiv", "Reddit"]
    methods = ["GCN", "GCN2", "GIN", "SAGE", "GGP", "RBF"]
    actions = ["GNN", "GP"]

    for name in dnames:
        if name.upper().startswith(args.data.upper()):
            print("Dataset: %s" % name)
            break
    else:
        message = "Unsupported data! Possible values: " + " ".join(dnames)
        raise Exception(message)

    for method in methods:
        if method.upper() == args.method.upper():
            args.method = method
            message = method
            if args.action == "GP":
                if method not in ["GGP", "RBF"]: message += "-GP" 
                if args.Nystrom: message += "-X"
            print("Method: %s" % message)
            break
    else:
        message = "Unsupported method! Possible values: " + " ".join(methods)
        raise Exception(message)

    if args.action not in actions:
        message = "Unsupported action! Possible values: " + " ".join(actions)
        raise Exception(message)

    transform = transition_matrix(normalization="sym" if method not in ["SAGE", "GGP"] else "row")
    data = load_data(name, center=args.center, scale=args.scale, transform=transform)
    device = torch.device("cuda:%s" % args.device if args.device>=0 and torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print("Dataset loaded to: %s" % device)

    if args.runs <= 0:
        if args.action == "GP" and args.fraction <= 1:
            print("Number of runs: 1 for deterministic results.")
            args.runs = 1
        else:
            print("Number of runs: 10 for random results.")
            args.runs = 10
    else:
        print("Number of runs: %d" % args.runs)
    if args.action == "GP" and args.Nystrom:
        print("Using Nystrom: 1/%d of training nodes chosen as landmark nodes." % args.fraction)
    print("----")

    if args.verbose:
        print("Preprocessing: center=%s, scale=%s" % (args.center, args.scale))
        if args.action == "GP":
            print("GP arguments:  num_layers=%d, sigma_b=%.2f, sigma_w=%.2f, Nystrom=%s, fraction=%d" % 
                  (args.num_layers, args.sigma_b, args.sigma_w, args.Nystrom, args.fraction))
        if args.action == "GNN":
            print("GNN arguments: num_layers=%d, dim_hidden=%d, dropout=%.2f, lr=%.4f, epochs=%d, batch_size=%d" %
                  (args.num_layers, args.dim_hidden, args.dropout, args.lr, args.epochs, args.batch_size))
        print("----")

    torch.manual_seed(123)

    def result_format(result):
        return "time: %.4f, train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2], result[3])
    
    if args.action == "GP":
        from time import process_time
        now = process_time()
        result_runs = torch.zeros((args.runs, 4))
        epsilon = torch.logspace(-3, 1, 101, device=device)
        params = {}
        if method == "GCN2": params = {"alpha":0.1, "theta":0.5}
        if method == "GGP": params = {"initial":"polynomial", "c":5.0, "d":3.0}
        model = GNNGP(data, device=device, Nystrom=args.Nystrom)
        for j in range(args.runs):
            if args.Nystrom:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            if method != "RBF":
                model.set_hyper_param(args.num_layers, args.sigma_b, args.sigma_w, method=method, **params)
                model.predict(epsilon)
                summary = model.get_summary()
                result_runs[j] = torch.tensor([process_time()-now, summary["train"], summary["val"], summary["test"]])
            else:
                for gamma in torch.logspace(-2, 2, 101):
                    model.set_hyper_param(initial="rbf", method=method, gamma=gamma)
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

    if args.action == "GNN":
        args.batchnorm = name in ["chameleon", "crocodile", "squirrel", "arxiv"]
        if args.batch_size <= 0:
            from main_gnn import main_gnn
            main_gnn(args, device, data, method)
        else:
            from main_batch import main_batch
            main_batch(args, device, data, method)

if __name__ == "__main__":
    main()
