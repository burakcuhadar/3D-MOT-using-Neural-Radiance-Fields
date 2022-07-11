import wandb


class LoggerWandb():
    def __init__(self, project_name, args):
        self.run = wandb.init(project=project_name)
        self.run.config.update(args)
        self.is_importance = args.N_importance > 0



    def log_train(self, epoch, loss, psnr, psnr0=None):
        #TODO tf.contrib.summary.histogram('tran', trans)
        if self.is_importance:
            self.run.log({"loss": loss, "psnr": psnr, "psnr0": psnr0, "epoch": epoch})
        else:
            self.run.log({"loss": loss, "psnr": psnr, "epoch": epoch})

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


    def log_test(self, loss, psnr, rgbs, disps):
        self.run.log({"test loss": loss, "test psnr": psnr})
        columns = [f'rgb{i}' for i in range(len(rgbs))]
        columns += [f'disp{i}' for i in range(len(rgbs))]
        test_table = wandb.Table(columns=columns)
        test_table.add_data(*rgbs, *disps)
        self.run.log({"test table": test_table})

    def log_video(self, rgb_path, disp_path):
        # Create video table
        columns = ["rgb", "disp"]
        video_table = wandb.Table(columns=columns)
        video_table.add_data(
            wandb.Video(rgb_path, fps=30, format="mp4"), 
            wandb.Video(disp_path, fps=30, format="mp4"))
        self.run.log({"video table": video_table})


