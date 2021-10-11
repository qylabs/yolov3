import re
import sys
import os
import torch
from torch.utils import data
import torchvision
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from reidtools import visualize_ranked_results


def euclidean_dist(x, y):
    """
    smaller dist means closer score
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
    larger dist means closer score
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


def eval_market1501(distmat, q_ids, g_ids, max_rank=10):
    """
    Evaluates CMC(Cumulative Matching Characteristics) rank and mAP
        Acc_k=1 if topk g_id==q_id else 0
        CMC used for closed_gallery_set tests
        
        AP=(num_of_topkg_id==q_id)/tot_query_number
        mAP=mean_of_all_q_id_AP
        recall=(num_of_topkg_id==q_id)/tot_gallery_g_id=q_id

    # Evaluation with market1501 metric
    # Key: for each query identity, its gallery images from the same camera view are discarded.

    distmat: np.array, shape (num_q, num_g) 2D  distmat smaller means closer
    q_ids: query label ids, 1D shape=num_q
    g_ids: gallery label ids, 1D
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_ids[indices] == q_ids[:, np.newaxis]).astype(np.int32)#This is to find which (row,col) is matched in the distmat

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    #loop for every q_id
    for q_idx in range(num_q):
        # get query id
        q_id = q_ids[q_idx]

        # compute cmc curve
        raw_cmc = matches[q_idx] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank]) #only obtain top rank-k cmc
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum() #tot g_id==q_id in the gallery
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)] #here precision is cumulative precision. different from classification precision correction_num/tot_num
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc #obtain only true query precision
        AP = tmp_cmc.sum() / num_rel #average of precision
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)#mean of tot query

    return all_cmc, all_AP, mAP



def load_data(data_dir):
    # data loader
    root = data_dir
    query_dir = os.path.join(root,"query")
    gallery_dir = os.path.join(root,"gallery")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128,64)),
        # torchvision.transforms.Resize((256, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    querydataset = torchvision.datasets.ImageFolder(query_dir, transform=transform)
    gallerydataset = torchvision.datasets.ImageFolder(gallery_dir, transform=transform)

    return querydataset,gallerydataset



features_file= sys.argv[1] #the arg1
print('features_file ',features_file)
features = torch.load(features_file)
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]



score_mat=cosin_dist(qf,gf) #smaller dist means larger score similarity
top5_similarity,top5_idx=score_mat.topk(5, dim=1)
res = top5_idx[:,0] #got top1
top1correct = gl[res].eq(ql).sum().item()
# top5correct=gl[top5_idx].eq(ql.unsqueeze(1)).sum().item()

print("===Summary===")
print("gallery feature/label",gf.size(),gl.size())
print("query feature/label",qf.size(),ql.size())

print('top1 gallery label=\n',gl[res])
print('query label=\n',ql)
print("Acc top1:{:.3f}".format(top1correct/ql.size(0))) #0.985

C2= confusion_matrix(gl[res], ql)
print("confusion matrix:")
print(C2)

# print('top5 cosin similarity=\n',top5_similarity[res])
# print('top5_idx=\n',top5_idx[res])
# print('top5_idx label=\n',gl[top5_idx[res]])
# print("Acc top5:{:.3f}".format(top5correct/ql.size(0)))


# dist_mat = qf.mm(gf.t())
# dist_mat=euclidean_dist(qf,gf).numpy()
dist_mat=1-cosin_dist(qf,gf).numpy() #q_num*g_num
print('dist_mat.shape ',dist_mat.shape)
q_ids=ql.numpy()
g_ids=gl.numpy()


all_cmc, all_AP, mAP=eval_market1501(dist_mat, q_ids, g_ids, max_rank=10)
print('all_cmc \n',all_cmc)
print('all_AP \n',all_AP,len(all_AP))
print('mAP: ',mAP)

title='cmc'+features_file.split('features')[-1].split('.')[0]
plt.title(title)
plt.plot(range(1,len(all_cmc)+1),all_cmc,marker='o',markersize=3)
plt.xlim(left=0.2)
plt.savefig(title+'.jpg')


#visualize ranked results
if sys.argv[2]:
    querydataset,gallerydataset=load_data(sys.argv[2])
    visualize_ranked_results(dist_mat,(querydataset.imgs,gallerydataset.imgs),
                            data_type='image',width=64,height=128,
                            save_dir='result',topk=5)
    
