# GCN
python3 src/main.py --data=cora --method=gcn --action=gnn
python3 src/main.py --data=cite --method=gcn --action=gnn
python3 src/main.py --data=pubm --method=gcn --action=gnn
python3 src/main.py --data=arxi --method=gcn --action=gnn
python3 src/main.py --data=cham --method=gcn --action=gnn --lr=0.3162277
python3 src/main.py --data=squi --method=gcn --action=gnn --lr=0.3162277
python3 src/main.py --data=croc --method=gcn --action=gnn --lr=0.3162277 --epochs=200
python3 src/main.py --data=redd --method=gcn --action=gnn --runs=10 --epochs=10 --verbose

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
python3 src/main.py --data=cham --method=gcn --action=gp --sigma_b=0.3162277 --fraction=1
python3 src/main.py --data=squi --method=gcn --action=gp --sigma_b=0.3162277 --fraction=1
python3 src/main.py --data=croc --method=gcn --action=gp --sigma_b=0.3162277 --fraction=1
python3 src/main.py --data=arxi --method=gcn --action=gp --fraction=50 --center
python3 src/main.py --data=redd --method=gcn --action=gp --fraction=50

# RBF and RBF-X
python3 src/main.py --data=cham --method=gcn --action=rbf
python3 src/main.py --data=cham --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=squi --method=gcn --action=rbf
python3 src/main.py --data=squi --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=croc --method=gcn --action=rbf
python3 src/main.py --data=croc --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=cora --method=gcn --action=rbf
python3 src/main.py --data=cora --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=cite --method=gcn --action=rbf
python3 src/main.py --data=cite --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=pubm --method=gcn --action=rbf
python3 src/main.py --data=pubm --method=gcn --action=rbf --fraction=1
python3 src/main.py --data=arxi --method=gcn --action=rbf --fraction=50 --runs=1
python3 src/main.py --data=redd --method=gcn --action=rbf --fraction=50 --runs=1