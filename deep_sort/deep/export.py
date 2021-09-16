import torch
import torch.nn as nn
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
    parser.add_argument('--img-channel', type=int, default=3, help='input img channel')  # support various img channel
    opt = parser.parse_args()
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    if args.model_name:
        print('use model: ',args.model_name)
        net=build_model(args.model_name,num_classes=num_classes, pretrained=True)
    else:
        net = Net(num_classes=num_classes)

    checkpoint = torch.load(args.checkpoint,map_location=device)
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    # Update model
    if opt.half:
        img, model = img.half(), model.half()  # to FP16
    
    for _ in range(2):
        y = model(img)  # dry runs
    
    # ONNX export ------------------------------------------------------------------------------------------------------
    f = opt.weights.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=False, opset_version=opt.opset_version, input_names=['images'],
                        training=torch.onnx.TrainingMode.TRAINING if opt.train else torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=not opt.train,
                    #   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                    #                 'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None
                        dynamic_axes={'images': {0: 'batch'}} if opt.dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model

    # Simplify
        if opt.simplify:
            import onnxsim
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=opt.dynamic,
                input_shapes={'images': list(img.shape)} if opt.dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)