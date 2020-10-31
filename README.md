# pytorch-fid-wrapper
A simple wrapper around @mseitzer's great pytorch-fid work.

```python
import  pytorch_fid_wrapper as pfw

# set pfw's configuration with your parameters
pfw.set_config(batch_size=BATCH_SIZE, dims=DIMS, device=DEVICE)

# compute real_m and real_s only once, they will not change during training
real_images = my_validation_data # N x C x H x W tensor
real_m, real_s = pfw.compute_real_val_stats(real_images)

# get the fake images your model currently generates
fake_images = my_model.compute_fake_images() # N x C x H x W tensor

# compute the fid score
val_fid = pfw.calculate_val_fid(fake_images, real_m, real_s)
```

Remember to cite their work if using the present wrapper or their code directly:

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
