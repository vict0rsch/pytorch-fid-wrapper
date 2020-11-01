import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pytorch-fid-wrapper",
    version="0.0.2",
    author="Victor Schmidt",
    author_email="not.an.address@yes.com",
    description=(
        "Wrapper around the pytorch-fid package to compute Frechet Inception"
        + "Distance (FID) using PyTorch in-memory given tensors of images."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vict0rsch/pytorch-fid-wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.5",
    install_requires=["numpy", "pillow", "scipy", "torch", "torchvision"],
)
