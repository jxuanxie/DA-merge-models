# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import os
from argparse import ArgumentParser
import numpy as np

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    # 检查初始化的device是否正确
    print(args.device)
    # test a single image
    result = inference_segmentor(model, args.img)

    # 打印np数组
    numpy_result = np.array(result)
    print(numpy_result.shape)
    numpy_result = numpy_result[0, :, :]
    print(numpy_result.shape)
    print(numpy_result)
    np.savetxt("./demo_image.txt", numpy_result, fmt='%d')
    arr1 = np.loadtxt("demo_image.txt", dtype=np.int32)
    print(arr1)
    # show the results
    file, extension = os.path.splitext(args.img)
    pred_file = f'{file}_pred{extension}'
    assert pred_file != args.img
    model.show_result(
        args.img,
        result,
        palette=get_palette(args.palette),
        out_file=pred_file,
        show=False,
        opacity=args.opacity)
    print('Save prediction to', pred_file)


if __name__ == '__main__':
    main()
