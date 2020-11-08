"""
# ----------------------------
# -----  pfw docstrings  -----
# ----------------------------

Adapted from:
https://github.com/mseitzer/pytorch-fid/blob/4d7695b39764ba1d54ab6639e0695e5c4e6f346a/pytorch_fid/fid_score.py

Modifications are:
  * modify calculate_activation_statistics ot handle in-memory N x C x H x W tensors
    instead of file lists with a dataloader
  * add fid() and get_stats()

# ---------------------------------------------
# -----  pytorch-fid original docstrings  -----
# ---------------------------------------------

Calculates the Frechet Inception Distance (FID) to evaluate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code adapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d


from pytorch_fid_wrapper.inception import InceptionV3
from pytorch_fid_wrapper import params as pfw_params


def get_activations(images, model, batch_size=50, dims=2048, device="cpu"):
    """
    Calculates the activations of the pool_3 layer for all images.

    Args:
        images ([type]): Tensor of images N x C x H x W
        model ([type]): Instance of inception model
        batch_size (int, optional): Batch size of images for the model to process at
            once. Make sure that the number of samples is a multiple of
            the batch size, otherwise some samples are ignored. This behavior is
            retained to match the original FID score implementation. Defaults to 50.
        dims (int, optional): Dimensionality of features returned by Inception.
            Defaults to 2048.
        device (str | torch.device, optional): Device to run calculations.
            Defaults to "cpu".

    Returns:
        np.ndarray: A numpy array of dimension (num images, dims) that contains the
            activations of the given tensor when feeding inception with the query
            tensor.
    """

    model.eval()

    n_batches = len(images) // batch_size

    assert n_batches > 0, (
        "Not enough images to make at least 1 full batch. "
        + "Provide more images or decrease batch_size"
    )

    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for b in range(n_batches):

        batch = images[b * batch_size : (b + 1) * batch_size].to(device)

        if batch.nelement() == 0:
            continue

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.ndarray): Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
        sigma1 (np.ndarray): The covariance matrix over activations for generated
            samples.
        mu2 (np.ndarray): The sample mean over activations, precalculated on a
            representative data set.
        sigma2 (np.ndarray): The covariance matrix over activations, precalculated on an
            representative data set.
        eps (float, optional): Fallback in case of infinite covariance.
            Defaults to 1e-6.

    Returns:
        float: The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    images, model, batch_size=50, dims=2048, device="cpu"
):
    """
    Calculation of the statistics used by the FID.

    Args:
        images (torch.Tensor): Tensor of images N x C x H x W
        model (torch.nn.Module): Instance of inception model
        batch_size (int, optional): The images tensor is split into batches with
            batch size batch_size. A reasonable batch size depends on the hardware.
            Defaults to 50.
        dims (int, optional): Dimensionality of features returned by Inception.
            Defaults to 2048.
        device (str | torch.device, optional): Device to run calculations.
            Defaults to "cpu".

    Returns:
        tuple(np.ndarray, np.ndarray): (mu, sigma)
            mu => The mean over samples of the activations of the pool_3 layer of
                the inception model.
            sigma => The covariance matrix of the activations of the pool_3 layer of
                the inception model.
    """
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_stats(images, model=None, batch_size=None, dims=None, device=None):
    """
    Get the InceptionV3 activation statistics (mu, sigma) for a batch of `images`.

    If `model` (InceptionV3) is not provided, it will be instanciated according
    to `dims`.

    Other arguments are optional and will be inherited from `pfw.params` if not
    provided. Use `pfw.set_config` to change those params globally for future calls


    Args:
        images (torch.Tensor): The images to compute the statistics for. Must be
            N x C x H x W
        model (torch.nn.Module, optional): InceptionV3 model. Defaults to None.
        batch_size (int, optional): Inception inference batch size.
            Will use `pfw.params.batch_size` if not provided. Defaults to None.
        dims (int, optional): which inception block to select. See
            InceptionV3.BLOCK_INDEX_BY_DIM. Will use pfw.params.dims if not provided.
            Defaults to None.
        device (str | torch.device, optional): PyTorch device for inception inference.
            Will use pfw.params.device if not provided. Defaults to None.

    Returns:
        tuple(np.ndarray, np.ndarray): (mu, sigma)
            mu => The mean over samples of the activations of the pool_3 layer of
                the inception model.
            sigma => The covariance matrix of the activations of the pool_3 layer of
                the inception model.
    """
    if batch_size is None:
        batch_size = pfw_params.batch_size
    if dims is None:
        dims = pfw_params.dims
    if device is None:
        device = pfw_params.device

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
    else:
        assert isinstance(model, InceptionV3)
    return calculate_activation_statistics(images, model, batch_size, dims, device)


def fid(
    fake_images,
    real_images=None,
    real_m=None,
    real_s=None,
    batch_size=None,
    dims=None,
    device=None,
):
    """
    Computes the FID score of `fake_images` w.r.t. either precomputed stats on real
    data, or another batch of images (typically real ones).

    If `real_images` is `None`, you must provide `real_m` **and** `real_s` with
    matching dimensions to `fake_images`.

    If `real_images` is not `None` it will prevail over `real_m` and `real_s`
    which will be ignored

    Other arguments are optional and will be inherited from `pfw.params` if not
    provided. Use `pfw.set_config` to change those params globally for future calls

    Args:
        fake_images (torch.Tensor): N x C x H x W tensor.
        real_images (torch.Tensor, optional): N x C x H x W tensor. If provided,
            stats will be computed from it, ignoring real_s and real_m.
            Defaults to None.
        real_m (, optional): Mean of a previous activation stats computation,
            typically on real data. Defaults to None.
        real_s (, optional): Std of a previous activation stats computation,
            typically on real data. Defaults to None.
        batch_size (int, optional): Inception inference batch_size.
            Will use pfw.params.batch_size if not provided. Defaults to None.
        dims (int, optional): which inception block to select.
            See InceptionV3.BLOCK_INDEX_BY_DIM. Will use pfw.params.dims
            if not provided. Defaults to None.
        device (str | torch.device, optional): PyTorch device for inception inference.
            Will use pfw.params.device if not provided. Defaults to None.

    Returns:
        float: Frechet Inception Distance between `fake_images` and either `real_images`
            or `(real_m, real_s)`
    """

    assert real_images is not None or (real_m is not None and real_s is not None)

    if batch_size is None:
        batch_size = pfw_params.batch_size
    if dims is None:
        dims = pfw_params.dims
    if device is None:
        device = pfw_params.device

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    if real_images is not None:
        real_m, real_s = get_stats(real_images, model, batch_size, dims, device)

    fake_m, fake_s = get_stats(fake_images, model, batch_size, dims, device)

    fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)
    return fid_value
