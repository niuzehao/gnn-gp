# ArXiv Dataset
python3 src/main.py arxi gcn gnn
python3 src/main.py arxi gcn2 gnn
python3 src/main.py arxi gin gnn
python3 src/main.py arxi sage gnn
python3 src/main.py arxi gcn gp  --fraction=50 --center
python3 src/main.py arxi gcn2 gp --fraction=50 --center
python3 src/main.py arxi gin gp  --fraction=50 --center
python3 src/main.py arxi sage gp --fraction=50 --center

# PubMed Dataset
python3 src/main.py pubm gcn gnn
python3 src/main.py pubm gcn2 gnn
python3 src/main.py pubm gin gnn
python3 src/main.py pubm sage gnn
python3 src/main.py pubm gcn gp
python3 src/main.py pubm gcn2 gp
python3 src/main.py pubm gin gp
python3 src/main.py pubm sage gp

# Reddit Dataset
python3 src/main.py redd gcn gp  --fraction=50
python3 src/main.py redd gcn2 gp --fraction=50
python3 src/main.py redd gin gp  --fraction=50
python3 src/main.py redd sage gp --fraction=50 --sigma_b=0.316
python3 src/main.py redd gcn gnn  --epochs=10 --batch_size=10240 --verbose
python3 src/main.py redd gcn2 gnn --epochs=10 --batch_size=10240 --verbose
python3 src/main.py redd gin gnn  --epochs=10 --batch_size=10240 --verbose
python3 src/main.py redd sage gnn --epochs=10 --batch_size=10240 --verbose