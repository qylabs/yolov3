import test
import yaml
from models.experimental import attempt_load
from utils.datasets import create_dataloader
import argparse
from utils.general import colorstr
import torch
DATA='data/WiderPerson_gray.yaml'
WEIGHTS='runs/train/exp9/weights/qlast.pt'
#WEIGHTS='runs/best_finetune_wt.pt'
SOURCE='mydataset/test_qy'
batch_size = 32
HYP='data/hyp.scratch_gray.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--hyp', type=str, default=HYP, help='hyperparameters path')
parser.add_argument('--single-cls', action='store_false', help='train multi-class data as single-class')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
#parser.add_argument('--gray_input', action='store_false', help='save model path')
opt = parser.parse_args()

with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

with open(DATA) as f:
        data_dict = yaml.safe_load(f)  # data dict

model = attempt_load(WEIGHTS, map_location='cpu')  # load FP32 model

#model = torch.load(WEIGHTS)
gs = max(int(model.stride.max()), 32)  # grid size (max stride)

testloader = create_dataloader(SOURCE, 320, batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       workers=0,
                                       pad=0.5, prefix=colorstr('val: '))[0]
print(testloader)
results, maps, times = test.test(data_dict,
                                conf_thres=0.5,
                                batch_size=batch_size,
                                imgsz=320,
                                model=model,
                                single_cls=opt.single_cls,
                                dataloader=testloader)