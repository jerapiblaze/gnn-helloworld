import torch
import torch.nn.functional as F
from dataset_milp.bg import Planetoid
from NN.gcn import GCN

def TrainModel(model, data, optimizer:torch.optim.Optimizer, n_epochs:int):
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"epoch_{epoch} loss={loss}")

    return model

def Main():
    dataset = Planetoid(root='./data/datasets', name='Cora')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    trained_model = TrainModel(model, data, optimizer, 1000)
    torch.save(trained_model, "./data/trained/dummy.pkl")

if __name__=="__main__":
    Main()