import datetime
import socket
import torch


def set_outdir_by_exp_name(cfg, mode='train'):
    ostr = cfg.work_dir
    ostr += f"_g{torch.cuda.device_count()}{datetime.datetime.now().strftime('%m%d%H%M')}_{socket.gethostname()[:4]}"
    cfg.work_dir = ostr
    return
