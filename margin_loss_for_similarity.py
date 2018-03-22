import torch
import numpy as np
from torch.autograd import Variable


def myfunc(a):
    # we should map 0 --> 1 and non-zero --> -1
    if a == 0.0:
        return 1.0
    else:
        return -1.0


def get_signs_matrix(labels1, labels2):
    labels1 = labels1.view(-1, 1)
    labels2 = labels2.view(-1, 1)
    n = labels1.size(0)
    d = labels2.size(1)

    x = labels1.unsqueeze(1).expand(n, n, d)
    y = labels2.unsqueeze(0).expand(n, n, d)

    distances_between_labels = torch.abs(x - y).sum(2)
    #print('distances_between_labels ', distances_between_labels)
    # we should map 0 --> 1 and non-zero --> -1
    vfunc = np.vectorize(myfunc)
    signs = torch.from_numpy(vfunc(distances_between_labels.data.cpu().numpy())).float().cuda()
    #print('signs ', signs)
    return Variable(signs)


class MarginLossForSimilarity(torch.nn.Module):
    def __init__(self, alpha=0.3, bethe=1.2):
        super(MarginLossForSimilarity, self).__init__()
        self.alpha = alpha
        self.bethe = bethe
        print('self.alpha = ', self.alpha)
        print('self.bethe = ', self.bethe)

    def forward(self,outputs, labels):
        """
        D_ij =  euclidean distance between representations x_i and x_j
        y_ij =  1 if x_i and x_j represent the same object
        y_ij = -1 otherwise

        margin(i, j) := (alpha + y_ij (D_ij − bethe))+
        {loss}        = (1/n) * sum_ij (margin(i, j))

        """
        distances_matrix = torch.mm(outputs, outputs.transpose(0, 1))
        signs_matrix = get_signs_matrix(labels, labels)
        n = float(distances_matrix.data.shape[0])
        #print('inputed distances matrix ', distances_matrix)
        #distances_matrix = distances_matrix.view(params.batch_size_for_similarity, params.batch_size_for_similarity)
        #print('after reshape distances matrix ', distances_matrix)
        margin = torch.clamp(self.alpha + signs_matrix * (distances_matrix - self.bethe), min=0.0).cuda()
        loss = torch.sum(margin)/n
        return loss
