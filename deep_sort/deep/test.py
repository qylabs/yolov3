import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net
from modellib import build_model

parser = argparse.ArgumentParser(description="Test on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument('--checkpoint', type=str, help="checkpoint path")
parser.add_argument('--save_feature', type=str,default="features.pth", help="save feature path")
parser.add_argument("--model-name",default='',type=str,help="model name")
parser.add_argument("--num_classes",type=int,help="model class number")
parser.add_argument("--in_channel",default=3,type=int,help="model input channel")
parser.add_argument("--galler_id_reform",default=2,type=int,help="galler_id_reform")
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=64, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=64, shuffle=False
)

# net definition
if args.model_name:
    print('use model: ',args.model_name)
    net=build_model(args.model_name,num_classes=args.num_classes, pretrained=True,in_channel=args.in_channel,reid=True)
else:
    net = Net(num_classes=args.num_classes,reid=True)

assert os.path.isfile(args.checkpoint), "Error: no checkpoint file found!"
print('Loading weight from ',args.checkpoint)
checkpoint = torch.load(args.checkpoint,map_location=device)
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(device)
print("Net\n",net)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx,(inputs,labels) in enumerate(queryloader):
        print('==query idx ',idx)
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))

    for idx,(inputs,labels) in enumerate(galleryloader):
        print('==gallery idx ',idx)
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels =gallery_labels-args.galler_id_reform #because gallery include other_not_included_person(-1) and background(0)

# save features
features = {
    "qf": query_features.squeeze(),
    "ql": query_labels,
    "gf": gallery_features.squeeze(),
    "gl": gallery_labels
}
torch.save(features,args.save_feature)