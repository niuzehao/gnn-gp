# GCN
python3 src/main.py --data=cora --method=gcn --action=gnn
python3 src/main.py --data=cite --method=gcn --action=gnn
python3 src/main.py --data=pubm --method=gcn --action=gnn
python3 src/main.py --data=cham --method=gcn --action=gnn --lr=0.3162277
python3 src/main.py --data=squi --method=gcn --action=gnn --lr=0.3162277
python3 src/main.py --data=croc --method=gcn --action=gnn --lr=0.3162277
python3 src/main.py --data=arxi --method=gcn --action=gnn
python3 src/main.py --data=redd --method=gcn --action=gnn --epochs=10 --verbose --batch_size=10240

# GCNGP and GCNGP-X
python3 src/main.py --data=cora --method=gcn --action=gp
python3 src/main.py --data=cite --method=gcn --action=gp
python3 src/main.py --data=pubm --method=gcn --action=gp
python3 src/main.py --data=cham --method=gcn --action=gp --sigma_b=0.3162277
python3 src/main.py --data=squi --method=gcn --action=gp --sigma_b=0.3162277
python3 src/main.py --data=croc --method=gcn --action=gp --sigma_b=0.3162277
python3 src/main.py --data=cora --method=gcn --action=gp --fraction=1
python3 src/main.py --data=cite --method=gcn --action=gp --fraction=1
python3 src/main.py --data=pubm --method=gcn --action=gp --fraction=1
python3 src/main.py --data=cham --method=gcn --action=gp --fraction=1 --sigma_b=0.3162277
python3 src/main.py --data=squi --method=gcn --action=gp --fraction=1 --sigma_b=0.3162277
python3 src/main.py --data=croc --method=gcn --action=gp --fraction=1 --sigma_b=0.3162277
python3 src/main.py --data=arxi --method=gcn --action=gp --fraction=50 --center
python3 src/main.py --data=redd --method=gcn --action=gp --fraction=50

# RBF and RBF-X
python3 src/main.py --data=cham --method=gcn --action=rbf
python3 src/main.py --data=squi --method=gcn --action=rbf
python3 src/main.py --data=croc --method=gcn --action=rbf
python3 src/main.py --data=cora --method=gcn --action=rbf
python3 src/main.py --data=cite --method=gcn --action=rbf
python3 src/main.py --data=pubm --method=gcn --action=rbf
python3 src/main.py --data=cham --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=squi --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=croc --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=cora --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=cite --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=pubm --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=arxi --method=gcn --action=rbf --fraction=50 --runs=1
python3 src/main.py --data=redd --method=gcn --action=rbf --fraction=50 --runs=1

# GGP and GGP-X
python3 src/main.py --data=cham --method=ggp --action=gp
python3 src/main.py --data=squi --method=ggp --action=gp
python3 src/main.py --data=croc --method=ggp --action=gp
python3 src/main.py --data=cora --method=ggp --action=gp
python3 src/main.py --data=cite --method=ggp --action=gp
python3 src/main.py --data=pubm --method=ggp --action=gp
python3 src/main.py --data=cham --method=ggp --action=gp --fraction=1
python3 src/main.py --data=squi --method=ggp --action=gp --fraction=1
python3 src/main.py --data=croc --method=ggp --action=gp --fraction=1
python3 src/main.py --data=cora --method=ggp --action=gp --fraction=1
python3 src/main.py --data=cite --method=ggp --action=gp --fraction=1
python3 src/main.py --data=pubm --method=ggp --action=gp --fraction=1
python3 src/main.py --data=arxi --method=ggp --action=gp --fraction=50 --runs=1
python3 src/main.py --data=redd --method=ggp --action=gp --fraction=50 --runs=1