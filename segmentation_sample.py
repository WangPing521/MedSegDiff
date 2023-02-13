
import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
from utils.datasets.UDA.mmwhs import CTInterface, MRInterface

sys.path.append("")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed =10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min )/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    # if args.data_name == 'ISIC':
    #     tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    #     transform_test = transforms.Compose(tran_list)
    #
    #     ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
    #     args.in_ch = 4
    # elif args.data_name == 'BRATS':
    #     tran_list = [transforms.Resize((args.image_size,args.image_size)),]
    #     transform_test = transforms.Compose(tran_list)
    #
    #     ds = BRATSDataset(args.data_dir,transform_test)
    #     args.in_ch = 5

    handler_ct = CTInterface()
    handler_mr = MRInterface()

    train_set, val_set, test_set = handler_ct._create_datasets()
    # train_set, val_set, test_set = handler_mr._create_datasets()

    datal = th.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False)
    args.in_ch = 2
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        # b, m, path = next(data)  #should return an image from the dataloader "data"
        # c = th.randn_like(b[:, :1, ...])

        val_data, path = next(data)
        b = val_data[0] # img 1,1,h,w
        c = th.randn_like(val_data[1].float()) # noise 1,1,h,w


        img = th.cat((b, c), dim=1)  # add a noise channel$
        slice_ID =path[0].split("_")[-1].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample

            co = th.tensor(cal_out)
            enslist.append(co)

            if args.debug:
                s = sample[: ,-1 ,: ,:].unsqueeze(1)
                o = org[: ,:-1 ,: ,:]
                c = cal

                tup = (o ,s ,c ,co)

                compose = th.cat(tup ,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, args.out_dir + str(slice_ID) + '_output' + str(i) + ".jpg"), nrow=1,
                                  padding=10)
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres, fp= os.path.join(args.out_dir, args.out_dir + str(slice_ID) + '_output_ens' + ".jpg"), nrow=1, padding=10)


def create_argparser():
    defaults = dict(
        data_name='MMWHS',
        data_dir="../dataset/MMWHS/training",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="results/savedmodel000000.pt",
        num_ensemble=5,  # number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results_infer/',
        multi_gpu=None,  # "0,1,2"
        debug=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
