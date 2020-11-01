"""
# ---------------------------------
# -----  pfw's global params  -----
# ---------------------------------

batch_size: batch_size for inception inference.
    Defaults to 50 as in pytorch-fid.
dims: which inception block to select from InceptionV3.BLOCK_INDEX_BY_DIM.
    Defaults to 2048 as in pytorch-fid.
device: PyTorch device for inception inference.
    Defaults to cpu as in pytorch-fid.
"""
batch_size = 50
dims = 2048
device = "cpu"
