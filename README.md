# pytorch-fid-wrapper
A simple wrapper around [@mseitzer](https://github.com/mseitzer)'s great [**pytorch-fid**](https://github.com/mseitzer/pytorch-fid) work.

The goal is to compute the Fréchet Inception Distance between two sets of images *in-memory* using PyTorch.

## Usage

```python
import  pytorch_fid_wrapper as pfw

# Optional: set pfw's configuration with your parameters once and for all
pfw.set_config(batch_size=BATCH_SIZE, dims=DIMS, device=DEVICE)

# compute real_m and real_s only once, they will not change during training
real_images = my_validation_data # N x C x H x W tensor
real_m, real_s = pfw.get_stats(real_images)

# get the fake images your model currently generates
fake_images = my_model.compute_fake_images() # N x C x H x W tensor

# compute the fid score
val_fid = pfw.fid(fake_images, real_m, real_s)
# OR
new_real_data = some_other_validation_data # N x C x H x W tensor
val_fid = pfw.fid(fake_images, new_real_data)
```

Please refer to [**pytorch-fid**](https://github.com/mseitzer/pytorch-fid) for any documentation on the InceptionV3 implementation or FID calculations.

## Config

`pfw.get_stats(...)` and `pfw.fid(...)` need to know what block of the InceptionV3 model to use (`dims`), on what device to compute inference (`device`) and with what batch size (`batch_size`).

Default values are in `pfw.params`: `batch_size = 50`, `dims = 2048` and `device = "cpu"`. If you want to override those, you have to options:

1/ override any of these parameters in the function calls. For instance:
  ```python
  pfw.fid(fake_images, new_real_data, device="cuda:0")
  ```
2/ override the params globally with `pfw.set_config` and set them for all future calls without passing parameters again. For instance:
  ```python
  pfw.set_config(batch_size=100, dims=768, device="cuda:0")
  ...
  pfw.fid(fake_images, new_real_data)
  ```

## Recognition

Remember to cite their work if using [`pytorch-fid-wrapper`](https://github.com/vict0rsch/pytorch-fid-wrapper) or [**pytorch-fid**](https://github.com/mseitzer/pytorch-fid):

```
@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.1.1},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
