# Best launch recipies

### August 21st
I find the number of samples + training steps is much larger than we expect.
```
uv run main.py \
    --n_images_per_class 6000 \
    --warmup_n_steps_train 40000 \
    --mstep_n_steps_train 5000
```
This has been giving the best performance so far. However, when the model architecture is DiT, you can see ghostly/residual pixelation on the samples, corresponding to the choice of `patchify` size. 
