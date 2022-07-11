import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils.logging import LoggerWandb
from torch.utils.data import DataLoader
from models.nerf import NeRF
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from models.rendering import render_path, render
from models.rendering import mse2psnr, img2mse, to8b
from datasets import dataset_dict
from utils.io import *


def get_scheduler(args, optimizer):
    if args.lrate_decay_steps:
        return MultiStepLR(optimizer, milestones=args.lrate_decay_steps, gamma=args.lrate_decay_rate)
    else:
        return StepLR(optimizer, step_size=args.lrate_decay, gamma=args.lrate_decay_rate)




# NOTE: static cam visualization is not implemented 
def train():

    parser = config_parser()
    args = parser.parse_args()

    # Create training dataset & loader 
    dataset_class = dataset_dict[args.dataset_type]
    train_dataset = dataset_class(args, split='train')
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = 1 if args.no_batching else args.N_rand,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_sample_pts_and_viewdirs,
        pin_memory=True)

    # Create validation dataset & loader
    val_dataset = dataset_class(args, split='val')
    val_dataloader = DataLoader (
        val_dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=1,
        collate_fn=val_dataset.collate_sample_pts_and_viewdirs,
        pin_memory=True)

    create_log_dir(args.basedir, args.expname)
    copy_config_save_args(args.basedir, args.expname, args)

    #Create models
    model_coarse = NeRF(D=args.netdepth, W=args.netwidth, args=args)
    model_coarse.to(device)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)
        model_fine.to(device)
    

    # Create optimizer
    grad_vars = list(model_coarse.parameters()) 
    if args.N_importance > 0:
        grad_vars += list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    scheduler = get_scheduler(args, optimizer)

    # Load checkpoint
    step_restored = 0
    if args.ckpt_path is not None:
        step_restored = load_ckpt(args.ckpt_path, model_coarse, model_fine, optimizer, scheduler)
        print("Resuming from step:", step_restored)
    else:
        print("No checkpoint given, starting from scratch")

    epochs = args.epochs
    
    logger_wandb = LoggerWandb(project_name=args.expname, args=args)

    print('use_batching', train_dataset.use_batching)
    print('number of batches', len(train_dataloader))
    
    for step in tqdm(range(step_restored+1, epochs+1), desc="Training epochs"):
        
        model_coarse.train()
        model_fine.train()
        train_loss_running = 0
        train_psnr_running = 0
        train_psnr0_running = 0
        
        # Training loop
        #for batch_i, batch in enumerate(train_dataloader):
        for batch in tqdm(train_dataloader, leave=False, desc=f"Epoch {step}"):
            
            train_dataset.move_batch_to_device(batch, device)
            
            pts, viewdirs, z_vals, rays_o, rays_d, target = batch
            
            optimizer.zero_grad()

            rgb, disp, acc, extras = render(model_coarse, model_fine, pts, viewdirs, z_vals, rays_o, rays_d, 
                retraw=True, N_importance=args.N_importance)
            
            # Compute loss
            img_loss = img2mse(rgb, target)
            # TODO log: trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            
            loss.backward()
            optimizer.step()
            
            train_loss_running += loss.item()
            train_psnr_running += psnr.item()
            if 'rgb0' in extras:
                train_psnr0_running += psnr0.item()
            
        
        scheduler.step()

        # Log loss and psnr
        if step % args.epoch_print == 0:
            avg_loss = train_loss_running / len(train_dataloader)
            avg_psnr = train_psnr_running / len(train_dataloader)
            avg_psnr0 = train_psnr0_running / len(train_dataloader)
            print(f'Epoch: {step}, Loss: {avg_loss}, PSNR: {avg_psnr}')
            logger_wandb.log_train(step, avg_loss, avg_psnr, avg_psnr0 if avg_psnr0 != 0 else None)        

        # Save checkpoint
        if step % args.epoch_ckpt == 0:
            path = os.path.join(args.basedir, args.expname, f'epoch_{step}.ckpt')
            save_ckpt(path, model_coarse, model_fine, optimizer, scheduler, step)
            #print(f"Saved checkpoint for epoch {step}")

        # Validation step, render one random view from validation set
        if step % args.epoch_val == 0:
            val_batch = next(iter(val_dataloader))
            val_dataset.move_batch_to_device(val_batch, device)
            pts, viewdirs, z_vals, rays_o, rays_d, target = val_batch
            model_coarse.eval()
            model_fine.eval()
            with torch.no_grad():
                rgb, disp, acc, extras = render(model_coarse, model_fine, pts, viewdirs, z_vals, rays_o, rays_d, 
                    retraw=True, N_importance=args.N_importance)
            
            val_mse = img2mse(rgb, target)
            psnr = mse2psnr(val_mse)
            rgb0, disp0, z_std = None, None, None
            if 'rgb0' in extras:
                rgb0, disp0, z_std = extras['rgb0'], extras['disp0'], extras['z_std']
                rgb0 = to8b(torch.reshape(rgb0, (val_dataset.H, val_dataset.W, 3)).cpu().detach().numpy())
                disp0 = to8b(torch.reshape(disp0, (val_dataset.H, val_dataset.W, 1)).cpu().detach().numpy())
                z_std = to8b(torch.reshape(z_std, (val_dataset.H, val_dataset.W, 1)).cpu().detach().numpy())
            
            rgb = to8b(torch.reshape(rgb, (val_dataset.H, val_dataset.W, 3)).cpu().detach().numpy())
            disp = to8b(torch.reshape(disp, (val_dataset.H, val_dataset.W, 1)).cpu().detach().numpy())
            acc = to8b(torch.reshape(acc, (val_dataset.H, val_dataset.W, 1)).cpu().detach().numpy())
            target = to8b(torch.reshape(target, (val_dataset.H, val_dataset.W, 3)).cpu().detach().numpy())
            

            logger_wandb.log_val(step, val_mse, psnr, rgb, target, disp, acc, rgb0,
                 disp0, z_std)
    
        
         
        


if __name__=='__main__':
    train()




