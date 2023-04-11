from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling_VS import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fname = args.data
    filename = f'./samples/single-coil/{fname}.npy'

    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.C13_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    img = torch.from_numpy(np.load(filename).astype(np.complex64))
    img = img.view(1, 1, 40, 40, 40, 4)
    # img = img.view(44, 44,4,40)
    img = img.to(config.device)

    mask = get_mask(img[:,:,:,:,0:1,0:1], img_size, batch_size,
                    type=args.mask_type,
                    acc_factor=args.acc_factor,
                    center_fraction=args.center_fraction)
    # mask = torch.zeros_like(img)
    # mask[[0:2:round(0.4*img_size)]] = 1

    ckpt_filename_T = f"./weights/checkpoint_T.pth"
    ckpt_filename = f"./weights/checkpoint_X.pth"
    # Construct VESDE from class defined in sde_lib.py (N = number of time points)
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    # Instantiate predictor object from the ReverseDiffusionPredictor class (from sampling.py)
    predictor = ReverseDiffusionPredictor
    # Instantiate corrector object from the LangevinCorrector class (from sampling.py)
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint for time dimensions
    score_model_T = mutils.create_model(config)
    ema_T = ExponentialMovingAverage(score_model_T.parameters(),
                                   decay=config.model.ema_rate)
    state_T = dict(step=0, model=score_model_T, ema=ema_T)
    state_T = restore_checkpoint(ckpt_filename_T, state_T, config.device, skip_sigma=False)
    ema_T.copy_to(score_model_T.parameters())
    
    # create model and load checkpoint for spatial dimensions
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=False)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/single-coil')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################

    pc_fouriercs = get_pc_fouriercs_RI(sde,
                                       predictor, corrector,
                                       inverse_scaler,
                                       snr=snr,
                                       n_steps=m,
                                       probability_flow=probability_flow,
                                       continuous=config.training.continuous,
                                       denoise=True)
    # fft
    print(mask.size())
    print(img.size())
    kspace = fft2(img)
    # undersampling
    mask = mask*0
    mask = torch.transpose(mask,2,3)
    under_kspace = kspace * mask
    under_img = ifft2(under_kspace)

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, score_model_T, scaler(under_img), mask, Fy=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################
    input = under_img.squeeze().cpu().detach().numpy()
    label = img.squeeze().cpu().detach().numpy()
    mask_sv = mask.squeeze().cpu().detach().numpy()

    np.save(str(save_root / 'input' / fname) + '.npy', input)
    np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
    np.save(str(save_root / 'label' / fname) + '.npy', label)
    plt.imsave(str(save_root / 'input' / fname) + '.png', np.abs(input), cmap='gray')
    plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')
    plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')
    plt.imsave(str(save_root / 'input' / (fname + '_mask')) + '.png', np.abs(mask_sv), cmap='gray')

    
    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    np.save(str(save_root / 'recon' / 'kspace') + '.npy', kspace.cpu().detach().numpy())
    np.save(str(save_root / 'recon' / 'under_kspace') + '.npy', under_kspace.cpu().detach().numpy())
    np.save(str(save_root / 'recon' / 'img') + '.npy', img.cpu().detach().numpy())
    np.save(str(save_root / 'recon' / 'mask') + '.npy', mask.cpu().detach().numpy())

    plt.imsave(str(save_root / 'recon' / fname) + '.png', np.abs(recon), cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d','partial_fourier'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()