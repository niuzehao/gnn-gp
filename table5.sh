# ArXiv Dataset
python3 src/main.py --data=arxi --method=gcn --action=gnn
python3 src/main.py --data=arxi --method=gcn2 --action=gnn
python3 src/main.py --data=arxi --method=gin --action=gnn
python3 src/main.py --data=arxi --method=sage --action=gnn
python3 src/main.py --data=arxi --method=gcn --action=gp --fraction=50 --center
python3 src/main.py --data=arxi --method=gcn2 --action=gp --fraction=50 --center
python3 src/main.py --data=arxi --method=gin --action=gp --fraction=50 --center
python3 src/main.py --data=arxi --method=sage --action=gp --fraction=50 --center

# PubMed Dataset
python3 src/main.py --data=pubm --method=gcn --action=gnn
python3 src/main.py --data=pubm --method=gcn2 --action=gnn
python3 src/main.py --data=pubm --method=gin --action=gnn
python3 src/main.py --data=pubm --method=sage --action=gnn
python3 src/main.py --data=pubm --method=gcn --action=gp
python3 src/main.py --data=pubm --method=gcn2 --action=gp
python3 src/main.py --data=pubm --method=gin --action=gp
python3 src/main.py --data=pubm --method=sage --action=gp

# Reddit Dataset
python3 src/main.py --data=redd --method=gcn --action=gp --fraction=50
python3 src/main.py --data=redd --method=gcn2 --action=gp --fraction=50
python3 src/main.py --data=redd --method=gin --action=gp --fraction=50
python3 src/main.py --data=redd --method=sage --action=gp --fraction=50 --sigma_b=0.3162277
python3 src/main.py --data=redd --method=gcn --action=gnn --epochs=10 --verbose --batch_size=10240
python3 src/main.py --data=redd --method=gcn2 --action=gnn --epochs=10 --verbose --batch_size=10240
python3 src/main.py --data=redd --method=gin --action=gnn --epochs=10 --verbose --batch_size=10240
python3 src/main.py --data=redd --method=sage --action=gnn --epochs=10 --verbose --batch_size=10240