# How to Tune the Parameters

## Jun 26 
Setup
- CryoET w/ 32 tilts (observations) + 5deg per tilt.
- Clean distribution $\pi = \delta_x$ where $x$ is a solid torus.
    - To approximate the torus with a point cloud, I uniformaly sample the volume of the object with $N$ points (via rejection sampling).
- Pretrain model using psuedoinverse
I did an initial sweep, for   

I find
- It seems GVP interpolant helps a lot. See [here](https://wandb.ai/clarkmiyamoto-new-york-university/toy3d-pc-scsi/runs/e63ab6w1/overview?nw=nwuserclarkmiyamoto).
    - Joan had this intuitiion this would be important, but he wasn't sure why...
- I think a slower EMA $\gamma = 0.999$ could help.
- Convergence is slow. Need around 100 EM steps, w/ 200 steps per EM loop. Which takes around 8 hours.
