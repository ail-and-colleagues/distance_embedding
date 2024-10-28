import torch
import torch.nn.functional as F

from torchinfo import summary

# class Dist_Loss(torch.nn.Module):
#     def __init__(self, node_)
#         super(Dist_Loss, self).__init__()

class embedding(torch.nn.Module):
    def __init__(self, dim_feature, num_k, w):
        super(embedding, self).__init__()
        self.dim_feature = dim_feature
        self.num_k = num_k

        try:
            self.weight = torch.nn.Parameter(data=torch.from_numpy(w.T).float())
        except:
            print('init embedding layer with random weights')
            self.weight = torch.nn.Parameter(torch.FloatTensor(self.num_k, self.dim_feature))
            torch.nn.init.xavier_uniform_(self.weight)

        print('weight: ', self.weight)

    def forward(self, x):
        x = F.linear(x, self.weight) 
        return x

class Dist2Pos(torch.nn.Module):
    def __init__(self, node_num, embed_dim, w=None):
        super(Dist2Pos, self).__init__()

        self.node_num = node_num
        self.embed_dim = embed_dim
        self.lin = embedding(self.node_num, self.embed_dim, w)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.lin(x)
        return x
    
    def fetch(self):
        w = self.lin.weight
        return w.to('cpu').detach().numpy().copy() 

if __name__ == "__main__":
    n = 66
    embed_dim = 2
    d2p = Dist2Pos(n, embed_dim)
    summary(d2p)