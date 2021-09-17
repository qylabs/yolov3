import argparse
import torch
import torch.nn as nn
import onnx
from model import Net
from modellib import build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='optimize TorchScript for mobile')  # TorchScript-only
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')  # ONNX-only
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')  # ONNX-only
    parser.add_argument('--opset-version', type=int, default=12, help='ONNX opset version')  # ONNX-only
    parser.add_argument("--model-name",default='',type=str,help="model name")
    parser.add_argument("--num_classes",type=int,help="model class number")
    parser.add_argument("--in_channel",default=3,type=int,help="model input channel")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    if args.model_name:
        print('use model: ',args.model_name)
        model=build_model(args.model_name,num_classes=args.num_classes, pretrained=True,in_channel=args.in_channel,reid=True)
    else:
        model = Net(num_classes=args.num_classes,reid=True)

    checkpoint = torch.load(args.weights,map_location=device)
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    model.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    print('best_acc: ',best_acc)

    img = torch.randn(args.batch_size, args.in_channel, *args.img_size).to(device)
    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    
    for _ in range(2):
        y = model(img)  # dry runs
    
    # ONNX export ------------------------------------------------------------------------------------------------------
    f = args.weights.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=False, opset_version=args.opset_version, input_names=['images'],
                        training=torch.onnx.TrainingMode.TRAINING if args.train else torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=not args.train,
                    #   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                    #                 'output': {0: 'batch', 2: 'y', 3: 'x'}} if args.dynamic else None
                        dynamic_axes={'images': {0: 'batch'}} if args.dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model

    # Simplify
    if args.simplify:
        import onnxsim
        print("==simplify==")
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=args.dynamic,
            input_shapes={'images': list(img.shape)} if args.dynamic else None)
        onnx.save(model_onnx, f)