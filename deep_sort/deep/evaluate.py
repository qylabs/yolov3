import re
import sys
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    # dist.addmm_(1, -2, x, y.t())
    dist+=-2*torch.mm(x,y.T)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosin_dist(x,y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    x_norm=x.div(x.norm(p=2,dim=1,keepdim=True))
    y_norm=y.div(y.norm(p=2,dim=1,keepdim=True))
    dist=torch.mm(x_norm,y_norm.T)
    return dist


features_file= sys.argv[1] #the arg1
print('features_file ',features_file)
features = torch.load(features_file)
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

# scores = qf.mm(gf.t())
# scores=euclidean_dist(qf,gf) #not work here for model Net
scores=cosin_dist(qf,gf) #3368*19732
top5_similarity,top5_idx=scores.topk(5, dim=1)
res = top5_idx[:,0] #got top1
top1correct = gl[res].eq(ql).sum().item()
# top5correct=gl[top5_idx].eq(ql.unsqueeze(1)).sum().item()

print("===Summary===")
print("gallery feature/label",gf.size(),gl.size())
print("query feature/label",qf.size(),ql.size())

print('top1 label=\n',gl[res])
print('query label=\n',ql)
print("Acc top1:{:.3f}".format(top1correct/ql.size(0))) #0.985

C2= confusion_matrix(gl[res], ql)
print("confusion matrix:")
# print('   0, 1, 2, 3')
print(C2)

# print('top5 cosin similarity=\n',top5_similarity[res])
# print('top5_idx=\n',top5_idx[res])
# print('top5_idx label=\n',gl[top5_idx[res]])
# print("Acc top5:{:.3f}".format(top5correct/ql.size(0)))

mask_false=torch.where(gl[res]!=ql)[0]

print()





