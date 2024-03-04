# main loop
'''
to run all the models, evaluate by metrics in metrics.py
python main.py # -m model_name -d dataset_name 
'''
from itertools import product

def run(m, d):
    pass

def evaluate(result):
    pass

def save_to_final(evaluation):
    '''append a row to final.csv'''
    pass

models = ['POP','S-POP','BPR-MF','Item-KNN','TIFU-KNN','STAN','GRU4Rec','NARM',
          'RepeatNet','SR-GNN','BERT4Rec','CORE','CoHHN','RepeatNet-DASP','Ride Buy-Cycle','MgRIA']
datasets = {'Equity':['recall@3','recall@10','mrr@10','ndcg@10'], 
            'Tafeng':['recall@10','recall@20','mrr@20','ndcg@20'],
            'Taobao':['recall@20','mrr@20','ndcg@20']
            }

for m, d in product(models, datasets):
    # do combination on each loop
    result = run(m, d)   # train and test each model on each dataset
    evaluation = evaluate(result)    # evaluate result by different metrics
    save_to_final(evaluation)
