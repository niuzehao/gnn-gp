# ArXiv Dataset
python3 src/main.py --data=arxi --method=gcn --action=gnn
python3 src/main.py --data=arxi --method=gcn2 --action=gnn
python3 src/main.py --data=arxi --method=gin --action=gnn
python3 src/main.py --data=arxi --method=sage --action=gnn
python3 src/main.py --data=arxi --method=gcn --action=gp --fraction=50 --center
python3 src/main.py --data=arxi --method=gcn2 --action=gp --fraction=50 --cent
python3 src/main.py --data=arxi --method=gin --action=gp --fraction=50 --center
python3 src/main.py --data=arxi --method=sage --action=gp --fraction=50 --center --sigma_b=0.3162277

# Reddit Dataset
python3 src/main.py --data=redd --method=gcn --action=gnn --runs=10 --epochs=10 --verbose
python3 src/main.py --data=redd --method=gcn2 --action=gnn --runs=10 --epochs=10 --verbose
python3 src/main.py --data=redd --method=gin --action=gnn --runs=10 --epochs=10 --verbose
python3 src/main.py --data=redd --method=sage --action=gnn --runs=10 --epochs=10 --verbose
python3 src/main.py --data=redd --method=gcn --action=gp --fraction=50
python3 src/main.py --data=redd --method=gcn2 --action=gp --fraction=50
python3 src/main.py --data=redd --method=gin --action=gp --fraction=50
python3 src/main.py --data=redd --method=sage --action=gp --fraction=100 --sigma_b=0.3162277