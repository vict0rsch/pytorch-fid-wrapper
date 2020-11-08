__version__ = "0.0.4"
from importlib import import_module
from pathlib import Path
from pytorch_fid_wrapper import params
from pytorch_fid_wrapper.fid_score import fid, get_stats
from pytorch_fid_wrapper.inception import InceptionV3


def set_config(batch_size=None, dims=None, device=None):
    """
    Sets pfw's global configuration to get rid of parameters in function
    calls when they don't change over the course of training.

    Any one of them can be set independently.

    Args:
        batch_size (int, optional): batch_size for inception inference.
            Defaults to None.
        dims (int, optional): which inception block to select.
            See InceptionV3.BLOCK_INDEX_BY_DIM. Defaults to None.
        device (any, optional): PyTorch device, as a string or device instance.
            Defaults to None.
    """
    if batch_size is not None:
        assert isinstance(batch_size, int)
        assert batch_size > 0
        params.batch_size = batch_size
    if dims is not None:
        assert isinstance(dims, int)
        assert dims in InceptionV3.BLOCK_INDEX_BY_DIM
        params.dims = dims
    if device is not None:
        params.device = device


__all__ = [
    import_module(f".{f.stem}", __package__)
    for f in Path(__file__).parent.glob("*.py")
    if "__" not in f.stem
]

del import_module, Path
