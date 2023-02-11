# GCN
python3 src/main.py cora gcn gnn
python3 src/main.py cite gcn gnn
python3 src/main.py pubm gcn gnn
python3 src/main.py cham gcn gnn --lr=0.316
python3 src/main.py squi gcn gnn --lr=0.316
python3 src/main.py croc gcn gnn --lr=0.316
python3 src/main.py arxi gcn gnn
python3 src/main.py redd gcn gnn --epochs=10 --batch_size=10240 --verbose

# GCNGP and GCNGP-X
python3 src/main.py cora gcn gp
python3 src/main.py cite gcn gp
python3 src/main.py pubm gcn gp
python3 src/main.py cham gcn gp --sigma_b=0.316
python3 src/main.py squi gcn gp --sigma_b=0.316
python3 src/main.py croc gcn gp --sigma_b=0.316
python3 src/main.py cora gcn gp --fraction=1
python3 src/main.py cite gcn gp --fraction=1
python3 src/main.py pubm gcn gp --fraction=1
python3 src/main.py cham gcn gp --fraction=1 --sigma_b=0.316
python3 src/main.py squi gcn gp --fraction=1 --sigma_b=0.316
python3 src/main.py croc gcn gp --fraction=1 --sigma_b=0.316
python3 src/main.py arxi gcn gp --fraction=50 --center
python3 src/main.py redd gcn gp --fraction=50

# RBF and RBF-X
python3 src/main.py cham rbf gp
python3 src/main.py squi rbf gp
python3 src/main.py croc rbf gp
python3 src/main.py cora rbf gp
python3 src/main.py cite rbf gp
python3 src/main.py pubm rbf gp
python3 src/main.py cham rbf gp --fraction=1
python3 src/main.py squi rbf gp --fraction=1
python3 src/main.py croc rbf gp --fraction=1
python3 src/main.py cora rbf gp --fraction=1
python3 src/main.py cite rbf gp --fraction=1
python3 src/main.py pubm rbf gp --fraction=1
python3 src/main.py arxi rbf gp --fraction=50 --runs=1
python3 src/main.py redd rbf gp --fraction=50 --runs=1

# GGP and GGP-X
python3 src/main.py cham ggp gp
python3 src/main.py squi ggp gp
python3 src/main.py croc ggp gp
python3 src/main.py cora ggp gp
python3 src/main.py cite ggp gp
python3 src/main.py pubm ggp gp
python3 src/main.py cham ggp gp --fraction=1
python3 src/main.py squi ggp gp --fraction=1
python3 src/main.py croc ggp gp --fraction=1
python3 src/main.py cora ggp gp --fraction=1
python3 src/main.py cite ggp gp --fraction=1
python3 src/main.py pubm ggp gp --fraction=1
python3 src/main.py arxi ggp gp --fraction=50 --runs=1
python3 src/main.py redd ggp gp --fraction=50 --runs=1