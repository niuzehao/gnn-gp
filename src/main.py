import torch

from gnngp import GNNGP
from datasets import load_data, transition_matrix

import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='GNNGP arguments')
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--action', required=True, choices=['gp', 'gnn', 'rbf'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)

    # preprocessing
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--train_size', type=int, default=0)

    # GP arguments
    parser.add_argument('--max_L', type=int, default=1)
    parser.add_argument('--nystrom', action='store_true')
    parser.add_argument('--fraction', type=int, default=1)

    # GNN arguments
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    print(args)

    dnames = ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'crocodile', 'squirrel', 'arxiv', 'Reddit']
    methods = ['GCN', 'GCN2', 'GIN', 'APPNP']

    for name in dnames:
        if name.lower().startswith(args.data.lower()):
            print("Dataset: %s" % name)
            break

    data = load_data(name, normalize_features=args.normalize_features,
                     transform=transition_matrix() if args.action == 'gp' else None)

    torch.manual_seed(123)

    device = torch.device('cuda:%s' % args.device if args.device>=0 else 'cpu')
    data = data.to(device)
    print("Dataset loaded to %s" % device)

    method = args.method.upper()
    if not (method in methods):
        raise Exception("Unsupported Method!")

    if args.train_size > 0:
        print("Random select %.0f percent of training points" % (100*args.train_size/torch.sum(data.train_mask)))
    if args.nystrom:
        print("Random select %.0f percent of landmark points" % (100/args.fraction))
    
    if args.action == 'gp':
        torch.manual_seed(123)
        now = time.process_time()
        result_runs = torch.zeros((args.runs, 3))

        model = GNNGP(data, args.max_L, 0.0, 1.0, device=args.device, Nystrom=args.nystrom)

        def result_format(result):
            return "train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2])

        for j in range(args.runs):
            if args.train_size > 0:
                model.mask["train"] = data.train_mask & (torch.rand(model.N, device=device) < args.train_size/torch.sum(data.train_mask))
            if args.nystrom:
                model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
            model.computed = False
            model.get_kernel(method=method)
            epsilon = torch.logspace(-3, 1, 101, device=device)
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

    # if args.gp:
    #     Nystrom = args.nystrom
    #     method = 'gpny' if Nystrom else 'gp'
    #     model = GNNGP(data)
    #     num_eps = 101; epsilon = torch.logspace(-5, 0, num_eps)
    #     best_info = {key:'' for key in loss}
    #     best_val_err = {key:float('inf') for key in loss}
    #     for sigma_w in torch.logspace(args.min_sigw, args.max_sigw, args.num_sigw):
    #         for sigma_b in torch.logspace(args.min_sigb, args.max_sigb, args.num_sigb):
    #             model.set_hyper_param(0, sigma_b, sigma_w)
    #             model.get_kernel()
    #             for L in range(args.max_L):
    #                 model.add_layer(Nystrom=Nystrom)
    #                 error = model.error(epsilon, loss)
    #                 for key in loss:
    #                     message = model.get_message(key)
    #                     f = open('%s_%s_%s.txt' % (dname, key, method), 'a')
    #                     print(message); print(message, file = f)
    #                     f.close()
    #                     i = torch.argmin(error[key]['val'])
    #                     if error[key]['val'][i] < best_val_err[key]:
    #                         best_info[key] = message
    #                         best_val_err[key] = error[key]['val'][i]
    #     for key in loss:
    #         f = open('%s_%s_%s.txt' % (dname, key, method), 'a')
    #         message = '===best result===\n'
    #         message += best_info[key]
    #         print(message); print(message, file = f)
    #         f.close()

    # if args.sqexp:
    #     method = 'sqexp'
    #     model = GNNGP(data)
    #     num_ell = 101; ell = torch.logspace(-1, 3, num_ell)
    #     train_err = {}; val_err = {}; test_err = {}
    #     for key in loss:
    #         train_err[key] = torch.zeros(num_ell)
    #         val_err[key] = torch.zeros(num_ell)
    #         test_err[key] = torch.zeros(num_ell)
    #     distm = torch.cdist(model.X, model.X)
    #     from predict import fit, error
    #     if model.y.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
    #         ya = torch.nn.functional.one_hot(model.y[model.train_mask])
    #     else:
    #         ya = model.y[model.train_mask]
    #     for i, e in enumerate(ell):
    #         K = torch.exp(-0.5*(distm/e)**2)
    #         yp = fit(K, ya, model.train_mask, eps=(1e-2,))
    #         err1, err2, err3 = error(yp, model.y, model.train_mask, model.val_mask, model.test_mask, loss)
    #         for key in loss:
    #             train_err[key][i] = err1[key]
    #             val_err[key][i] = err2[key]
    #             test_err[key][i] = err3[key]
    #     for key in loss:
    #         i = torch.argmin(val_err[key])
    #         # f = open('%s_%s_%s.txt' % (dname, key, method), 'a')
    #         message = '===best result===\n'
    #         message += 'metric=%s, epsilon=%05.4f, train=%05.4f, val=%05.4f, test=%05.4f\n' % \
    #                     (key, 1e-2, train_err[key][i], val_err[key][i], test_err[key][i])
    #         print(message); # print(message, file = f)
    #         # f.close()


    # if args.gp and args.param_select:
    #     model = GNNGP(data, args.max_L, 0.0, 1.0, device=args.device, Nystrom=args.nystrom)
    #     model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
    #     X = model.X; A = model.A
    #     X -= torch.mean(X, axis=0).view((1,-1))
    #     epsilon = torch.logspace(-3, 1, 101, device=device)
    #     best_info = {key:'' for key in loss}
    #     best_val_err = {key:float('inf') for key in loss}
    #     for sigma_w in torch.logspace(-1, 1, 21):
    #         for sigma_b in torch.logspace(-2, -1, 6):
    #             model.set_hyper_param(0, sigma_b, sigma_w)
    #             model.get_kernel(method=method)
    #             for L in range(3):
    #                 model.add_layer(method=method)
    #                 error = model.get_error(epsilon)
    #                 message = model.get_message()
    #                 print(message)
    #                 i = torch.argmin(error['mr']['val'])
    #                 if error['mr']['val'][i] < best_val_err['mr']:
    #                     best_info['mr'] = message
    #                     best_val_err['mr'] = error['mr']['val'][i]
    #     print(best_info['mr'])


    if args.action == 'rbf':
        model = GNNGP(data, args.max_L, 0.0, 1.0, device=args.device, Nystrom=args.nystrom)
        model.mask["landmark"] = model.mask["train"] & (torch.rand(model.N, device=device) < 1/args.fraction)
        X = model.X; A = model.A
        X -= torch.mean(X, axis=0).view((1,-1))
        model.set_init_kernel()
        if args.nystrom: model.Q = model.Q0
        else: model.K = model.K0
        model.computed = True
        epsilon = torch.logspace(-3, 1, 101, device=device)
        error = model.get_error(epsilon)
        print(model.log)


    if args.action == 'gnn':
        from main_gnn import gnn_mr
        gnn_mr(args, device, data, method)

if __name__ == "__main__":
    main()
