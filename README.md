# pytorch-fid-wrapper
A simple wrapper around [@mseitzer](https://github.com/mseitzer)'s great [**pytorch-fid**](https://github.com/mseitzer/pytorch-fid) work.

The goal is to compute the Fréchet Inception Distance between two sets of images *in-memory* using PyTorch.

## Installation

[![PyPI](https://img.shields.io/pypi/v/pytorch-fid-wrapper.svg)](https://pypi.org/project/pytorch-fid/)

```
pip install pytorch-fid-wrapper
```

Requires (and will install) (as `pytorch-fid`):
  * Python >= 3.5
  * Pillow
  * Numpy
  * Scipy
  * Torch
  * Torchvision

## Usage

```python
import  pytorch_fid_wrapper as pfw

# ---------------------------
# -----  Initial Setup  -----
# ---------------------------

# Optional: set pfw's configuration with your parameters once and for all
pfw.set_config(batch_size=BATCH_SIZE, dims=DIMS, device=DEVICE)

# Optional: compute real_m and real_s only once, they will not change during training
real_m, real_s = pfw.get_stats(real_images)

...

# -------------------------------------
# -----  Computing the FID Score  -----
# -------------------------------------

val_fid = pfw.fid(fake_images, real_m=real_m, real_s=real_s) # (1)

# OR

val_fid = pfw.fid(fake_images, real_images=new_real_images) # (2)
```

All `_images` variables in the example above are `torch.Tensor` instances with shape `N x C x H x W`. They will be sent to the appropriate device depending on what you ask for (see [Config](#config)).

To compute the FID score between your fake images and some real dataset, you can **either** re-use pre-computed stats `real_m`, `real_s` at each validation stage `(1)`, **or** provide another dataset for which the stats will be computed (in addition to your fake images' which are computed in both scenarios) `(2)`. Score is computed in `pfw.fid_score.calculate_frechet_distance(...)`, following [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid)'s implementation.

Please refer to [**pytorch-fid**](https://github.com/mseitzer/pytorch-fid) for any documentation on the InceptionV3 implementation or FID calculations.

## Config

`pfw.get_stats(...)` and `pfw.fid(...)` need to know what block of the InceptionV3 model to use (`dims`), on what device to compute inference (`device`) and with what batch size (`batch_size`).

Default values are in `pfw.params`: `batch_size = 50`, `dims = 2048` and `device = "cpu"`. If you want to override those, you have two options:

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

Remember to cite their work if using [`pytorch-fid-wrapper`](https://github.com/vict0rsch/pytorch-fid-wrapper) or [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid):

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
