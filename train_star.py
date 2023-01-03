import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from utils.logging import LoggerWandb
from utils.io import set_seeds
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.star import STaR
from tqdm import tqdm, trange
import wandb

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

    return nerf_schedule

def get_pose_metrics(poses, gt_poses):
    gt_translation = gt_poses[:, :3]
    gt_rotation = gt_poses[:, 3:]
    translation = poses[:, :3]
    rotation = poses[:, 3:]
    
    trans_error = torch.mean(torch.sum((translation - gt_translation)**2, dim=1).sqrt())
    rot_error = torch.sum(torch.abs(rotation - gt_rotation), dim=1).mean()
    return trans_error, rot_error
    
'''
def compute_pose_grad(transformed_pts_coarse, transformed_pts_fine, transformed_pts_grad):
    """
    transformed_pts: (N_rays*N_samples, 3)
    """
    #print(transformed_pts.grad)
    with torch.no_grad():
        """
        transformed_pts_coarse_flat = torch.reshape(transformed_pts_coarse, (-1, 3))
        transformed_pts_fine_flat = torch.reshape(transformed_pts_fine, (-1, 3))
        transformed_pts_flat = torch.cat([transformed_pts_coarse_flat, transformed_pts_fine_flat], dim=0)
        """
        transformed_pts_flat = torch.reshape(transformed_pts_fine, (-1,3))    
        transformed_pts_grad_flat = torch.reshape(transformed_pts_grad, (-1, 3))

        d_epsilon = torch.zeros((transformed_pts_flat.shape[0], 3, 6), device=device)
        d_epsilon[:, 0, 0] = 1.
        d_epsilon[:, 1, 1] = 1.
        d_epsilon[:, 2, 2] = 1.
        d_epsilon[:, 0, 4] = transformed_pts_flat[:,2]
        d_epsilon[:, 0, 5] = -transformed_pts_flat[:,1]
        d_epsilon[:, 1, 3] = -transformed_pts_flat[:,2]
        d_epsilon[:, 1, 5] = transformed_pts_flat[:,0]
        d_epsilon[:, 2, 3] = transformed_pts_flat[:,1]
        d_epsilon[:, 2, 4] = -transformed_pts_flat[:,0]

        #transformed_pts.grad.data.zero_()
        #transformed_pts_grad_flat_norm = transformed_pts_grad_flat.norm(dim=1)
        #transformed_pts_grad_flat = transformed_pts_grad_flat[transformed_pts_grad_flat_norm > 1e-5]
        #d_epsilon = d_epsilon[transformed_pts_grad_flat_norm > 1e-5]
        #print('transformed pts avg grad norm', transformed_pts_grad_flat.norm(dim=1).mean())
        #print('transformed pts max grad norm', torch.max(transformed_pts_grad_flat.norm(dim=1)))
        #print('transformed pts min grad norm', torch.min(transformed_pts_grad_flat.norm(dim=1)))
        #print('num greater than 1e-5', transformed_pts_grad_flat_norm[transformed_pts_grad_flat_norm > 1e-5].numel())

        pose_grad = torch.einsum('na,nab->nb', transformed_pts_grad_flat, d_epsilon)
        pose_grad = torch.mean(pose_grad, axis=0)
        
        #print('pose grad norm', pose_grad.norm())
        



    return pose_grad
'''


def val_step(val_dataset, val_dataloader, train_render_dataset, train_render_dataloader, star_model, args, step, logger):
    
    val_H = val_dataset.H
    val_W = val_dataset.W

    val_batch = next(iter(val_dataloader))
    val_dataset.move_batch_to_device(val_batch, device)
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
    
    logger.log_val_online(step, val_mse.item(), psnr.item(), rgb, target, rgb_static, rgb_dynamic, disp, acc, rgb0, 
        rgb_static0, rgb_dynamic0, disp0, z_std)

    # render one random view with random vehicle pose from train dataset
    train_W = train_render_dataset.W
    train_H = train_render_dataset.H
    train_batch = next(iter(train_render_dataloader))
    train_render_dataset.move_batch_to_device(train_batch, device)
    pts, viewdirs, z_vals, rays_o, rays_d, target, frames = train_batch
    star_model.eval()
    with torch.no_grad():
        rgb, disp, acc, extras, _ = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
            frames=frames, retraw=True, N_importance=args.N_importance)
    
    rgb0, disp0, z_std = None, None, None
    if 'rgb0' in extras:
        rgb0, disp0, z_std = extras['rgb0'], extras['disp0'], extras['z_std']
        rgb0 = to8b(torch.reshape(rgb0, (train_H, train_W, 3)).cpu().detach().numpy())
        disp0 = to8b(torch.reshape(disp0, (train_H, train_W, 1)).cpu().detach().numpy())
        z_std = to8b(torch.reshape(z_std, (train_H, train_W, 1)).cpu().detach().numpy())
    
    rgb = to8b(torch.reshape(rgb, (train_H, train_W, 3)).cpu().detach().numpy())
    rgb_static = to8b(torch.reshape(extras['rgb_map_static'], (train_H, train_W, 3)).cpu().detach().numpy())
    rgb_dynamic = to8b(torch.reshape(extras['rgb_map_dynamic'], (train_H, train_W, 3)).cpu().detach().numpy())
    rgb_static0 = to8b(torch.reshape(extras['rgb_map_static0'], (train_H, train_W, 3)).cpu().detach().numpy())
    rgb_dynamic0 = to8b(torch.reshape(extras['rgb_map_dynamic0'], (train_H, train_W, 3)).cpu().detach().numpy())
    disp = to8b(torch.reshape(disp, (train_H, train_W, 1)).cpu().detach().numpy())
    acc = to8b(torch.reshape(acc, (train_H, train_W, 1)).cpu().detach().numpy())
    target = to8b(torch.reshape(target, (train_H, train_W, 3)).cpu().detach().numpy())
    
    logger.log_train_render(step, rgb, target, rgb_static, rgb_dynamic, disp, acc, rgb0, rgb_static0, rgb_dynamic0, 
        disp0, z_std)
    

def setup_dataset(dataset_class, args, k):
    train_online_dataset = dataset_class(args, split='train_online', num_frames=k)
    train_online_dataloader = DataLoader(
        train_online_dataset, 
        batch_size = 1 if args.no_batching else args.N_rand,
        num_workers=args.num_workers,
        collate_fn=train_online_dataset.collate_sample_pts_and_viewdirs,
        shuffle=True,
        pin_memory=True)    
    
    val_online_dataset = dataset_class(args, split='val_online', num_frames=k)
    val_online_dataloader = DataLoader (
        val_online_dataset,
        batch_size=1, 
        num_workers=1,
        collate_fn=val_online_dataset.collate_sample_pts_and_viewdirs,
        shuffle=True,
        pin_memory=True)

    train_render_dataset = dataset_class(args, split='train_render', num_frames=k)
    train_render_dataloader = DataLoader(
        train_render_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=train_render_dataset.collate_sample_pts_and_viewdirs,
        shuffle=True,
        pin_memory=True
    )
    
    return train_online_dataset, train_online_dataloader, val_online_dataset, val_online_dataloader, \
        train_render_dataset, train_render_dataloader


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

    
    step_restored = 0
    if args.appearance_ckpt_path is not None:
        step_restored = load_ckpt(args.appearance_ckpt_path, star_model, optimizer, scheduler)
        print("Resuming from step:", step_restored)
    else:
        print("No checkpoint given, starting from scratch")

    print('use_batching', train_appearance_dataset.use_batching)
    print('number of batches', len(train_appearance_dataloader))
    epochs = args.epochs_appearance

    m1 = args.appearance_init_thres

    scaler = torch.cuda.amp.GradScaler(enabled=False)

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

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
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

            #loss.backward()
            scaler.scale(loss).backward()

            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

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

load_gt_poses = False #TODO!!!!!!!!!!!

def train_online(star_model, args, logger_wandb):

    k = args.initial_num_frames

    # Create optimizer
    optimizer = torch.optim.Adam(star_model.get_nerf_params(), lr=args.lrate, betas=(0.9, 0.999))
    pose_optimizer = torch.optim.Adam([star_model.poses_], lr=args.lrate_pose)

    scheduler = LambdaLR(optimizer, lr_lambda=get_scheduler_online(args))

    # Load checkpoint for online training
    step_restored = 0
    if args.online_ckpt_path is not None:
        step_restored, k = load_ckpt_online(args.online_ckpt_path, star_model, optimizer, scheduler, pose_optimizer=pose_optimizer) 
        #step_restored, k = load_ckpt_online(args.online_ckpt_path, star_model, optimizer, scheduler) 
        print("Resuming online training from step:", step_restored)
    else:
        print("No checkpoint given for online training, starting from scratch")

    
    # Create training dataset/loader for online training  
    dataset_class = dataset_dict[args.dataset_type]
    train_online_dataset, train_online_dataloader, val_online_dataset, val_online_dataloader, train_render_dataset, \
        train_render_dataloader = setup_dataset(dataset_class, args, k)
    
    ''' noisy pose initialization'''
    if step_restored == 0:
        print('poses before assigning', star_model.poses_)
        with torch.no_grad():
            star_model.poses_ += train_online_dataset.get_noisy_gt_relative_poses()[1:,...].to(device)
    

    print('starting online training with these poses: ', star_model.get_poses())

    if load_gt_poses:
        with torch.no_grad():
            star_model.gt_poses = train_online_dataset.gt_relative_poses_matrices.clone().to(device)

    print('use_batching', train_online_dataset.use_batching)
    print('number of batches', len(train_online_dataloader))
    epochs = args.epochs_online

    m2 = args.online_thres

    scaler = torch.cuda.amp.GradScaler(enabled=False) #TODO enabled=True

    for step in tqdm(range(step_restored+1, epochs+1), desc="Online Training epochs"):

        if step != step_restored+1 and train_fine_loss_running / len(train_online_dataloader) < m2:
            path = os.path.join(args.basedir, args.expname + '_' + logger_wandb.run.id, f'online_epoch_{step}.ckpt')
            save_ckpt_star_online(path, star_model, optimizer, scheduler, step, k)
            
            print('poses for k=', k)
            print(star_model.get_poses())

            # visualize one of val views
            val_step(val_online_dataset, val_online_dataloader, train_render_dataset, train_render_dataloader, 
                star_model, args, step, logger_wandb)
            
            if k == 15: # TODO num_frames from args
                break            
            print('Incrementing k to:', k+1, 'at step', step)
            k += 1
            
            train_online_dataset, train_online_dataloader, val_online_dataset, val_online_dataloader, \
                train_render_dataset, train_render_dataloader = setup_dataset(dataset_class, args, k)

            with torch.no_grad():
                new_poses = star_model.poses_.detach().clone() 
                new_poses[k-2,:] = star_model.poses_[k-3,:].detach().clone()
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

            optimizer.zero_grad(set_to_none=True)
            if not load_gt_poses:
                pose_optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False): #TODO enabled=true
                rgb, disp, acc, extras, entropy = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
                    frames=frames, retraw=True, N_importance=args.N_importance)

                # Compute loss
                img_loss = img2mse(rgb, target)
                psnr = mse2psnr(img_loss)
                img_loss0 = img2mse(extras['rgb0'], target)
                loss = img_loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
                loss += args.entropy_weight * (entropy + extras['entropy0']) #TODO entropy!
        
            #loss.backward()
            scaler.scale(loss).backward()

            '''
            if frames[0,0] != 0:
                with torch.no_grad():
                    pose_grad = compute_pose_grad(extras['transformed_pts0'], extras['transformed_pts'], star_model.get_poses_grad())
                    star_model.poses_[frames[0,0]-1, :] += args.lrate_pose * pose_grad 
            '''

            #optimizer.step()
            scaler.step(optimizer)
            if not load_gt_poses:
                #pose_optimizer.step()
                scaler.step(pose_optimizer)
            
            scaler.update()

            train_fine_loss_running += img_loss.float().item()
            train_loss_running += loss.float().item()
            train_psnr_running += psnr.float().item()
            if 'rgb0' in extras:
                train_psnr0_running += psnr0.float().item()


        '''if load_gt_poses:
            with torch.no_grad():
                star_model.gt_poses = train_online_dataset.gt_relative_poses_matrices.clone().to(device)'''
            
        
        scheduler.step()

        # Log loss and psnr
        if step % args.epoch_print == 0:
            avg_fine_loss = train_fine_loss_running / len(train_online_dataloader) 
            avg_loss = train_loss_running / len(train_online_dataloader)
            avg_psnr = train_psnr_running / len(train_online_dataloader)
            avg_psnr0 = train_psnr0_running / len(train_online_dataloader)
            tqdm.write(f'Epoch: {step}, Avg Fine Loss: {avg_fine_loss}, Avg Loss: {avg_loss}, PSNR: {avg_psnr}')
            with torch.no_grad():
                trans_error, rot_error = get_pose_metrics(star_model.get_poses()[1:k,...], 
                    train_online_dataset.gt_relative_poses.to(device)[1:k,...])
            #logger_wandb.log_train_online(step, avg_fine_loss, avg_psnr, avg_psnr0 if avg_psnr0 != 0 else None, trans_error, rot_error, pose_grad.norm().item())        
            logger_wandb.log_train_online(step, avg_fine_loss, avg_psnr, avg_psnr0 if avg_psnr0 != 0 else None, trans_error, rot_error)
            tqdm.write(f'Poses: {star_model.get_poses()}')
            
 
        # Save checkpoint
        if step % args.epoch_ckpt == 0:
            path = os.path.join(args.basedir, args.expname + '_' + logger_wandb.run.id, f'online_epoch_{step}.ckpt')
            save_ckpt_star_online(path, star_model, optimizer, scheduler, step, k, pose_optimizer=pose_optimizer)

        # Validation step, render one random view from validation set
        if step % args.epoch_val == 0:
            # Optimized poses
            #print(star_model.poses_)
            val_step(val_online_dataset, val_online_dataloader, train_render_dataset, train_render_dataloader, 
                star_model, args, step, logger_wandb)
         




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

    wandb.watch(star_model, log='all')

    if args.skip_appearance_init:
        if args.appearance_ckpt_path is not None:
            load_ckpt_appearance(args.appearance_ckpt_path, star_model, device)
            print("Appearance initialization skipped and ckpt loaded.")
        elif args.online_ckpt_path is not None:
            print("Online training checkpoint provided, skipping appearance initialization and loading that checkpoint...")
        else:
            raise RuntimeError('Checkpoint for appearance initialization should be provided if it is skipped')
    else:
        train_appearance_init(star_model, args, logger_wandb)
        print("Appearance initialization finished.")


    print("Starting online training...")
    
    train_online(star_model, args, logger_wandb)

    print("Finished online training.")


if __name__=='__main__':
    set_seeds()
    train()




