import torch
import torch.nn.functional as F
from dataset_milp.bg import Planetoid
from NN.gcn import GCN

def EvaluateModel(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

def Main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='./data/datasets', name='Cora')
    data = dataset[0].to(device)
    loaded_model = torch.load("./data/trained/dummy.pkl")
    EvaluateModel(loaded_model, data)

if __name__=="__main__":
    Main()