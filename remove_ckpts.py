import os
import shutil

path = "/home/stud/cuhadar/storage/user/star_trans_only_lightning/3D-MOT-using-Neural-Radiance-Fields/sbatch/ckpts/appinit"
skip_dirs = ["s7l0s0tf", "s7o7h9ls", "3uzu10x3", "l1avwstm", "zi0faekg", "mj4z8phs", "pmf8wmzk", "hf3eovzu", "nh0qyl6d"]

with os.scandir(path) as it:
    for entry in it:
        # If in skip dirs, skip removing this dir
        if entry.name not in skip_dirs:
            print("deleting...", entry.name)
            shutil.rmtree(entry.path)
        """
        with os.scandir(entry.path) as ckptdir_it:
            for ckpt in ckptdir_it:
                pass
        """