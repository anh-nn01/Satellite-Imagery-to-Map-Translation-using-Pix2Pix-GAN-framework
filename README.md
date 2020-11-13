# Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework
PyTorch Implementation of Pix2Pix framework from scratch to train a U-Net with Generative Adversarial Network which translates Satellite Imagery to an equivalent Map.<br>
<b>Reference</b>: https://arxiv.org/abs/1611.07004

# Trained Generator and Discriminator:<br>
Click this link to download the trained weights for the Sat2Map Generator and Discriminator: [Download Weights](https://drive.google.com/file/d/1vvv2dXL98_M4SrjUgGps2vt1FzGRKH7B/view?usp=sharing)

# Hyper-parameters
As suggested by the paper in the reference, here are the values of the hyper-parameters to train the Sat2Map model:</br>
* Batch size: **1**
* Input and Output image size: **256 x 256**
* Learning rate: **0.0002**
* Momentum: [β1, β2] = **[0.5, 0.999]**
* λ_L1 = 100

# Ideas and Intuition of cGAN (conditional GAN)

# Generator Architecture - U-Net

# Discriminator Architecture - Convolutional Neural Network

# Loss Function

# Results
