import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from utils.logging import LoggerWandb
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.star import STaR
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from models.rendering import render_path, render_star
from models.rendering import mse2psnr, img2mse, to8b
from datasets import dataset_dict
from utils.io import *


def get_scheduler(args, optimizer):
    if args.lrate_decay_steps:
        return MultiStepLR(optimizer, milestones=args.lrate_decay_steps, gamma=args.lrate_decay_rate)
    else:
        return StepLR(optimizer, step_size=args.lrate_decay, gamma=args.lrate_decay_rate)

def get_scheduler_online(args):
    pose_schedule = lambda epoch: 1. # pose params lr is not decayed
    if args.lrate_decay_steps:
        # MultiStepLR
        def nerf_schedule(epoch):
            factor = 1.
            for step in args.lrate_decay_steps:
                if epoch < step:
                    break
                factor *= args.lrate_decay_rate
            return factor
    else:
        # StepLR
        def nerf_schedule(epoch):
            return args.lrate_decay_rate ** (epoch // args.lrate_decay)

    return [pose_schedule, nerf_schedule]

def train_appearance_init(star_model, args, logger_wandb):
    # Create training dataset/loader for appearance initialization  
    dataset_class = dataset_dict[args.dataset_type]
    train_appearance_dataset = dataset_class(args, split='train_appearance', num_frames=1)
    train_appearance_dataloader = DataLoader(
        train_appearance_dataset, 
        batch_size = 1 if args.no_batching else args.N_rand,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=train_appearance_dataset.collate_sample_pts_and_viewdirs,
        pin_memory=True)

    # Create validation dataset/loader for appearance initialization
    val_appearance_dataset = dataset_class(args, split='val_appearance', num_frames=1)
    val_appearance_dataloader = DataLoader (
        val_appearance_dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=1,
        collate_fn=val_appearance_dataset.collate_sample_pts_and_viewdirs,
        pin_memory=True)
    
    val_H = val_appearance_dataset.H
    val_W = val_appearance_dataset.W

    star_model.poses_.requires_grad = False

    # Create optimizer
    grad_vars = list(star_model.parameters()) 
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    scheduler = get_scheduler(args, optimizer)

    #TODO: Load checkpoint for appearance initialization
    step_restored = 0
    '''if args.ckpt_path is not None:
        step_restored = load_ckpt(args.ckpt_path, model_coarse, model_fine, optimizer, scheduler)
        print("Resuming from step:", step_restored)
    else:
        print("No checkpoint given, starting from scratch")'''

    print('use_batching', train_appearance_dataset.use_batching)
    print('number of batches', len(train_appearance_dataloader))
    epochs = args.epochs_appearance

    m1 = args.appearance_init_thres

    for step in tqdm(range(step_restored+1, epochs+1), desc="Appearance Initialization Training epochs"):

        if step != step_restored+1 and train_fine_loss_running / len(train_appearance_dataloader) < m1:
            print(f"Threshold m1 is reached at epoch {step}, exitting appearance init training...")
            path = os.path.join(args.basedir, args.expname + '_' + logger_wandb.run.id, f'appearance_epoch_{step}.ckpt')
            save_ckpt_star(path, star_model, optimizer, scheduler, step)
            break

        star_model.train()
        train_fine_loss_running = 0
        train_loss_running = 0
        train_psnr_running = 0
        train_psnr0_running = 0
        
        # Appearance Initialization Training loop
        for batch in tqdm(train_appearance_dataloader, leave=False, desc=f"Epoch {step}"):
            
            train_appearance_dataset.move_batch_to_device(batch, device)
            
            pts, viewdirs, z_vals, rays_o, rays_d, target, _ = batch
            
            optimizer.zero_grad()

            rgb, disp, acc, extras = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, frames=None, 
                retraw=True, N_importance=args.N_importance, appearance_init=True)
            
            # Compute loss
            img_loss = img2mse(rgb, target)
            loss = img_loss
            psnr = mse2psnr(img_loss)
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            #loss += args.entropy_weight * entropy        #TODO try with entropy

            loss.backward()
            optimizer.step()
            
            train_fine_loss_running += img_loss.item()
            train_loss_running += loss.item()
            train_psnr_running += psnr.item()
            if 'rgb0' in extras:
                train_psnr0_running += psnr0.item()
            
        
        scheduler.step()

        # Log loss and psnr
        if step % args.epoch_print == 0:
            avg_loss = train_loss_running / len(train_appearance_dataloader)
            avg_psnr = train_psnr_running / len(train_appearance_dataloader)
            avg_psnr0 = train_psnr0_running / len(train_appearance_dataloader)
            print(f'Epoch: {step}, Loss: {avg_loss}, PSNR: {avg_psnr}')
            logger_wandb.log_train_appearance(step, avg_loss, avg_psnr, avg_psnr0 if avg_psnr0 != 0 else None)        

        # Save checkpoint
        if step % args.epoch_ckpt == 0:
            path = os.path.join(args.basedir, args.expname + '_' + logger_wandb.run.id, f'appearance_epoch_{step}.ckpt')
            save_ckpt_star(path, star_model, optimizer, scheduler, step)

        # Validation step, render one random view from validation set
        if step % args.epoch_val == 0:
            val_batch = next(iter(val_appearance_dataloader))
            val_appearance_dataset.move_batch_to_device(val_batch, device)
            pts, viewdirs, z_vals, rays_o, rays_d, target, _ = val_batch
            star_model.eval()
            with torch.no_grad():
                rgb, disp, acc, extras = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
                    frames=None, retraw=True, N_importance=args.N_importance, appearance_init=True)
            
            val_mse = img2mse(rgb, target)
            psnr = mse2psnr(val_mse)
            rgb0, disp0, z_std = None, None, None
            if 'rgb0' in extras:
                rgb0, disp0, z_std = extras['rgb0'], extras['disp0'], extras['z_std']
                rgb0 = to8b(torch.reshape(rgb0, (val_H, val_W, 3)).cpu().detach().numpy())
                disp0 = to8b(torch.reshape(disp0, (val_H, val_W, 1)).cpu().detach().numpy())
                z_std = to8b(torch.reshape(z_std, (val_H, val_W, 1)).cpu().detach().numpy())
            
            rgb = to8b(torch.reshape(rgb, (val_H, val_W, 3)).cpu().detach().numpy())
            disp = to8b(torch.reshape(disp, (val_H, val_W, 1)).cpu().detach().numpy())
            acc = to8b(torch.reshape(acc, (val_H, val_W, 1)).cpu().detach().numpy())
            target = to8b(torch.reshape(target, (val_H, val_W, 3)).cpu().detach().numpy())
            
            logger_wandb.log_val_appearance(step, val_mse, psnr, rgb, target, disp, acc, rgb0,
                 disp0, z_std)
    

    star_model.poses_.requires_grad = True


def train_online(star_model, args, logger_wandb):

    k = args.initial_num_frames

    # Create training dataset/loader for online training  
    dataset_class = dataset_dict[args.dataset_type]
    train_online_dataset = dataset_class(args, split='train_online', num_frames=k)
    train_num_views = len(train_online_dataset.imgs)
    train_H = train_online_dataset.H
    train_W = train_online_dataset.W
    
    train_online_dataloader = DataLoader(
        train_online_dataset, 
        batch_size = 1 if args.no_batching else args.N_rand,
        #sampler=SubsetRandomSampler(range(train_num_views * k * train_H * train_W)),
        num_workers=args.num_workers,
        collate_fn=train_online_dataset.collate_sample_pts_and_viewdirs,
        shuffle=True,
        pin_memory=True)

    # Create validation dataset/loader for online training
    val_online_dataset = dataset_class(args, split='val_online', num_frames=k)
    val_num_views = len(val_online_dataset.imgs)
    val_H = val_online_dataset.H
    val_W = val_online_dataset.W
    
    val_online_dataloader = DataLoader (
        val_online_dataset,
        batch_size=1, 
        #sampler=SubsetRandomSampler(range(val_num_views * k * val_H * val_W)), 
        num_workers=1,
        collate_fn=val_online_dataset.collate_sample_pts_and_viewdirs,
        shuffle=True,
        pin_memory=True)
    
    star_model.gt_poses = train_online_dataset.get_gt_vehicle_poses(args).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': star_model.poses_, 'lr': args.lrate_pose}, 
        {'params': star_model.get_nerf_params()}
    ], lr=args.lrate, betas=(0.9, 0.999))
    
    '''print('optimizer devices')
    for group in optimizer.param_groups:
        for p in group['params']:
            print(p.device)
            if p.grad is not None:
                print(p.grad.device)
                if p.grad.device != device:
                    print(p)'''

    scheduler = LambdaLR(optimizer, lr_lambda=get_scheduler_online(args))

    #TODO: Load checkpoint for online training
    step_restored = 0
    '''if args.ckpt_path is not None:
        step_restored = load_ckpt(args.ckpt_path, model_coarse, model_fine, optimizer, scheduler)
        print("Resuming from step:", step_restored)
    else:
        print("No checkpoint given, starting from scratch")'''

    print('use_batching', train_online_dataset.use_batching)
    print('number of batches', len(train_online_dataloader))
    epochs = args.epochs_online

    m2 = args.online_thres

    for step in tqdm(range(step_restored+1, epochs+1), desc="Online Training epochs"):

        if step != step_restored+1 and train_fine_loss_running / len(train_online_dataloader) < m2:
            print('Incrementing k to:', k+1)
            k += 1
            train_online_dataset = dataset_class(args, split='train_online', num_frames=k)
            val_online_dataset = dataset_class(args, split='val_online', num_frames=k)
            train_online_dataloader = DataLoader(
                train_online_dataset, 
                batch_size = 1 if args.no_batching else args.N_rand,
                #sampler=SubsetRandomSampler(range(train_num_views * k * train_H * train_W)),
                num_workers=args.num_workers,
                collate_fn=train_online_dataset.collate_sample_pts_and_viewdirs,
                shuffle=True,
                pin_memory=True)
            val_online_dataloader = DataLoader (
                val_online_dataset,
                batch_size=1, 
                #sampler=SubsetRandomSampler(range(val_num_views * k * val_H * val_W)), 
                num_workers=1,
                collate_fn=val_online_dataset.collate_sample_pts_and_viewdirs,
                shuffle=True,
                pin_memory=True)
            with torch.no_grad():
                new_poses = star_model.poses_.detach().clone()
                new_poses[k,:] = star_model.poses_[k-1,:].detach().clone()
                star_model.poses_.copy_(new_poses)


        star_model.train()
        train_fine_loss_running = 0
        train_loss_running = 0
        train_psnr_running = 0
        train_psnr0_running = 0
        
        # Online Training loop
        for batch in tqdm(train_online_dataloader, leave=False, desc=f"Epoch {step}"):
            
            train_online_dataset.move_batch_to_device(batch, device)
            
            pts, viewdirs, z_vals, rays_o, rays_d, target, frames = batch
            
            optimizer.zero_grad()

            rgb, disp, acc, extras, entropy = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
                frames=frames, retraw=True, N_importance=args.N_importance)
            

            # Compute loss
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            img_loss0 = img2mse(extras['rgb0'], target)
            loss = img_loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
            #loss += args.entropy_weight * (entropy + extras['entropy0']) # TODO - or +, also try with no entropy regularization
            

            loss.backward()

            '''print('optimizer devices')
            for group in optimizer.param_groups:
                for p in group['params']:
                    print(p.device)
                    if p.grad is not None:
                        print(p.grad.device)
                        if p.grad.device != device:
                            print(p)'''

            optimizer.step()

            
            
            train_fine_loss_running += img_loss.item()
            train_loss_running += loss.item()
            train_psnr_running += psnr.item()
            if 'rgb0' in extras:
                train_psnr0_running += psnr0.item()
            
        
        scheduler.step()

        # Log loss and psnr
        if step % args.epoch_print == 0:
            avg_loss = train_fine_loss_running / len(train_online_dataloader) 
            avg_psnr = train_psnr_running / len(train_online_dataloader)
            avg_psnr0 = train_psnr0_running / len(train_online_dataloader)
            print(f'Epoch: {step}, Loss: {avg_loss}, PSNR: {avg_psnr}')
            logger_wandb.log_train_online(step, avg_loss, avg_psnr, avg_psnr0 if avg_psnr0 != 0 else None)        

        # Save checkpoint
        if step % args.epoch_ckpt == 0:
            path = os.path.join(args.basedir, args.expname + '_' + logger_wandb.run.id, f'online_epoch_{step}.ckpt')
            save_ckpt_star(path, star_model, optimizer, scheduler, step)

        # Validation step, render one random view from validation set
        if step % args.epoch_val == 0:
            # Optimized poses
            #print(star_model.poses_)

            val_batch = next(iter(val_online_dataloader))
            val_online_dataset.move_batch_to_device(val_batch, device)
            pts, viewdirs, z_vals, rays_o, rays_d, target, frames = val_batch
            star_model.eval()
            with torch.no_grad():
                rgb, disp, acc, extras, _ = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
                    frames=frames, retraw=True, N_importance=args.N_importance)
            
            val_mse = img2mse(rgb, target)
            psnr = mse2psnr(val_mse)
            rgb0, disp0, z_std = None, None, None
            if 'rgb0' in extras:
                rgb0, disp0, z_std = extras['rgb0'], extras['disp0'], extras['z_std']
                rgb0 = to8b(torch.reshape(rgb0, (val_H, val_W, 3)).cpu().detach().numpy())
                disp0 = to8b(torch.reshape(disp0, (val_H, val_W, 1)).cpu().detach().numpy())
                z_std = to8b(torch.reshape(z_std, (val_H, val_W, 1)).cpu().detach().numpy())
            
            rgb = to8b(torch.reshape(rgb, (val_H, val_W, 3)).cpu().detach().numpy())
            rgb_static = to8b(torch.reshape(extras['rgb_map_static'], (val_H, val_W, 3)).cpu().detach().numpy())
            rgb_dynamic = to8b(torch.reshape(extras['rgb_map_dynamic'], (val_H, val_W, 3)).cpu().detach().numpy())
            rgb_static0 = to8b(torch.reshape(extras['rgb_map_static0'], (val_H, val_W, 3)).cpu().detach().numpy())
            rgb_dynamic0 = to8b(torch.reshape(extras['rgb_map_dynamic0'], (val_H, val_W, 3)).cpu().detach().numpy())
            disp = to8b(torch.reshape(disp, (val_H, val_W, 1)).cpu().detach().numpy())
            acc = to8b(torch.reshape(acc, (val_H, val_W, 1)).cpu().detach().numpy())
            target = to8b(torch.reshape(target, (val_H, val_W, 3)).cpu().detach().numpy())
            
            logger_wandb.log_val_online(step, val_mse, psnr, rgb, target, rgb_static, rgb_dynamic, disp, acc, rgb0, 
                rgb_static0, rgb_dynamic0, disp0, z_std)
    




''' 
    1. appearance initialization
    2. joint optimization of static and dynamic nerf on first k frames:
        - k=k0
        - if MSE of fine model in first k frames < m2: k += 1. repeat.
'''
def train():
    #torch.cuda.set_sync_debug_mode('warn')
    parser = config_parser()
    args = parser.parse_args()

    logger_wandb = LoggerWandb(project_name=args.expname, args=args)
    create_log_dir(args.basedir, args.expname + '_' + logger_wandb.run.id)
    copy_config_save_args(args.basedir, args.expname + '_' + logger_wandb.run.id, args)

    print('decay steps', args.lrate_decay_steps)

    #Create model
    star_model = STaR(num_frames=15, args=args) # TODO num_frames from args
    star_model.to(device)

    if args.skip_appearance_init:
        if args.appearance_ckpt_path is not None:
            load_ckpt_appearance(args.appearance_ckpt_path, star_model, device)
            print("Appearance initialization skipped and ckpt loaded.")
        else:
            raise RuntimeError('Checkpoint for appearance initialization should be provided if it is skipped')
    else:
        train_appearance_init(star_model, args, logger_wandb)
        print("Appearance initialization finished.")

    print(star_model.get_poses())

    print("Starting online training...")
    
    train_online(star_model, args, logger_wandb)

    print("Finished online training.")


if __name__=='__main__':
    train()



