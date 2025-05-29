

# Just follow the install scripts of CapeFormer.

cd cape_extloc/cmods/deform_ops/
sh make.sh

# Train
CUDA_VISIBLE_DEVICES=3 python train.py --config cape_extloc/cfg/s1_e200.py

# Test about 78+ mPCK
CUDA_VISIBLE_DEVICES=3 python test.py cape_extloc/cfg/s1.py <pretrained_weights.pth>
