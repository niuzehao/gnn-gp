import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

from argparse import ArgumentParser
from time import process_time

def main():
    parser = ArgumentParser(description="GNNGP arguments")
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--action", required=True, choices=["gnn", "gp"])
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

    dnames = ["Cora", "CiteSeer", "PubMed", "chameleon", "crocodile", "squirrel", "arxiv", "Reddit"]
    methods = ["GCN", "GCN2", "GIN", "SAGE", "GGP", "RBF"]

    for name in dnames:
        if name.lower().startswith(args.data.lower()):
            print("Dataset: %s" % name)
            break
    else:
        message = "Unsupported data! Possible values: " + " ".join(dnames)
        raise Exception(message)

    for method in methods:
        if method.lower() == args.method.lower():
            message = method
            if args.action == "gp":
                if method not in ["GGP", "RBF"]: message += "-GP" 
                if args.Nystrom: message += "-X"
            print("Method: %s" % message)
            break
    else:
        message = "Unsupported method! Possible values: " + " ".join(methods)
        raise Exception(message)

    if args.verbose:
        print("Preprocessing: center=%s, scale=%s" % (args.center, args.scale))
        if args.action == "gp":
            print("GP arguments: num_layers=%d, sigma_b=%.2f, sigma_w=%.2f, Nystrom=%s, fraction=%d" % 
                  (args.num_layers, args.sigma_b, args.sigma_w, args.Nystrom, args.fraction))
        if args.action == "gnn":
            print("GNN arguments: num_layers=%d, dim_hidden=%d, dropout=%.2f, lr=%.4f, epochs=%d, batch_size=%d" %
                  (args.num_layers, args.dim_hidden, args.dropout, args.lr, args.epochs, args.batchnorm))

    transform = transition_matrix(normalization_in="sym" if method not in ["SAGE", "GGP"] else "row")
    data = load_data(name, center=args.center, scale=args.scale, transform=transform)
    device = torch.device("cuda:%s" % args.device if args.device>=0 and torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print("Dataset loaded to %s" % device)

    torch.manual_seed(123)
    if args.Nystrom:
        print("Using Nystrom. Randomly select %.0f percent of landmark points." % (100/args.fraction))
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
        params = {}
        if method == "GCN2": params.update({"alpha":0.1, "theta":0.5})
        if method == "GGP": params.update({"initial":"polynomial", "c":5.0, "d":3.0})
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

    if args.action == "gnn":
        args.batchnorm = name in ["chameleon", "crocodile", "squirrel", "arxiv"]
        if args.batch_size <= 0:
            from main_gnn import main_gnn
            main_gnn(args, device, data, method)
        else:
            from main_batch import main_batch
            main_batch(args, device, data, method)

if __name__ == "__main__":
    main()
