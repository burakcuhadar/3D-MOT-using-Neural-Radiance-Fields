import wandb
import os


class LoggerWandb():
    def __init__(self, project_name, args):
        self.run = wandb.init(project=project_name)
        self.run.config.update(args)
        self.is_importance = args.N_importance > 0


    # For vanilla NeRF
    def log_train(self, epoch, loss, psnr, psnr0=None):
        if self.is_importance:
            self.run.log({"loss": loss, "psnr": psnr, "psnr0": psnr0, "epoch": epoch})
        else:
            self.run.log({"loss": loss, "psnr": psnr, "epoch": epoch})

    # For Star, appearance init
    def log_train_appearance(self, epoch, loss, psnr, psnr0=None):
        self.run.log({"loss appearance": loss, "psnr appearance": psnr, "psnr0 appearance": psnr0, "epoch": epoch})

    # For Star, online training
    def log_train_online(self, epoch, loss, psnr, psnr0=None, trans_error=None, rot_error=None, pose_grad_avg_norm=None):
        self.run.log({"fine loss online": loss, "psnr online": psnr, "psnr0 online": psnr0, 
            "translation error": trans_error, "rotation error": rot_error, 
            "pose grad avg norm": pose_grad_avg_norm, "epoch": epoch})

    # For vanilla NeRF
    def log_val(self, epoch, loss, psnr, rgb, gt_rgb, disp, acc, rgb0=None, disp0=None, z_std=None):
        # Create val table
        columns = ["epoch", "rgb", "gt rgb", "disp", "acc"]
        if self.is_importance:
            columns += ["rgb_coarse", "disp_coarse", "z_std"]
        val_table = wandb.Table(columns=columns)

        if self.is_importance:
            val_table.add_data(epoch, wandb.Image(rgb), wandb.Image(gt_rgb), wandb.Image(disp), wandb.Image(acc), 
                wandb.Image(rgb0), wandb.Image(disp0), wandb.Image(z_std))
        else:
            val_table.add_data(epoch, wandb.Image(rgb), wandb.Image(gt_rgb), wandb.Image(disp), wandb.Image(acc))
        
        self.run.log({"val loss": loss, "val psnr": psnr, "epoch": epoch})
        self.run.log({"val table": val_table, "epoch": epoch})


    # For Star, appearance init
    def log_val_appearance(self, epoch, loss, psnr, rgb, gt_rgb, disp, acc, rgb0=None, disp0=None, z_std=None):
        # Create val table
        columns = ["epoch", "rgb", "gt rgb", "disp", "acc", "rgb_coarse", "disp_coarse", "z_std"]
        val_table = wandb.Table(columns=columns)

        val_table.add_data(epoch, wandb.Image(rgb), wandb.Image(gt_rgb), wandb.Image(disp), wandb.Image(acc), 
            wandb.Image(rgb0), wandb.Image(disp0), wandb.Image(z_std))
    
        self.run.log({"val appearance loss": loss, "val appearance psnr": psnr, "epoch": epoch})
        self.run.log({"val appearance table": val_table, "epoch": epoch})

    # For Star, online training
    def log_val_online(self, epoch, loss, psnr, rgb, gt_rgb, rgb_static, rgb_dynamic, disp, acc, rgb0=None, rgb_static0=None, rgb_dynamic0=None, disp0=None, z_std=None):
        # Create val table
        columns = ["epoch", "rgb", "gt rgb", "rgb static", "rgb dynamic", "disp", "acc", "rgb_coarse", "rgb static coarse", "rgb dynamic coarse", "disp_coarse", "z_std"]
        val_table = wandb.Table(columns=columns)

        val_table.add_data(epoch, wandb.Image(rgb), wandb.Image(gt_rgb), wandb.Image(rgb_static), 
            wandb.Image(rgb_dynamic), wandb.Image(disp), wandb.Image(acc), wandb.Image(rgb0), wandb.Image(rgb_static0), 
            wandb.Image(rgb_dynamic0), wandb.Image(disp0), wandb.Image(z_std))
    
        self.run.log({"val online loss": loss, "val online psnr": psnr, "epoch": epoch})
        self.run.log({"val online table": val_table, "epoch": epoch})

    def log_train_render(self, epoch, rgb, gt_rgb, rgb_static, rgb_dynamic, disp, acc, rgb0=None, rgb_static0=None, 
        rgb_dynamic0=None, disp0=None, z_std=None):
        
        # Create table
        columns = ["epoch", "rgb", "gt rgb", "rgb static", "rgb dynamic", "disp", "acc", "rgb_coarse", "rgb static coarse", "rgb dynamic coarse", "disp_coarse", "z_std"]
        train_table = wandb.Table(columns=columns)

        train_table.add_data(epoch, wandb.Image(rgb), wandb.Image(gt_rgb), wandb.Image(rgb_static), 
            wandb.Image(rgb_dynamic), wandb.Image(disp), wandb.Image(acc), wandb.Image(rgb0), wandb.Image(rgb_static0), 
            wandb.Image(rgb_dynamic0), wandb.Image(disp0), wandb.Image(z_std))
        
        self.run.log({"train view render": train_table, "epoch": epoch})

    # For vanilla NeRF
    def log_test(self, loss, psnr, rgbs, disps):
        self.run.log({"test loss": loss, "test psnr": psnr})
        columns = [f'rgb{i}' for i in range(len(rgbs))]
        columns += [f'disp{i}' for i in range(len(rgbs))]
        test_table = wandb.Table(columns=columns)
        test_table.add_data(*rgbs, *disps)
        self.run.log({"test table": test_table})

    # For vanilla NeRF
    def log_video(self, rgb_path, disp_path, rgb_static_path, rgb_dynamic_path):
        # Create video table
        columns = ["rgb", "disp", "rgb static", "rgb dynamic"]
        video_table = wandb.Table(columns=columns)
        video_table.add_data(
            wandb.Video(rgb_path, fps=30, format="mp4"), 
            wandb.Video(disp_path, fps=30, format="mp4"),
            wandb.Video(rgb_static_path, fps=30, format="mp4"),
            wandb.Video(rgb_dynamic_path, fps=30, format="mp4"))
        self.run.log({"video table": video_table})


