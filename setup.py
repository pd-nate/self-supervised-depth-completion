from setuptools import find_packages, setup

setup(
    name="ssdc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "Pillow",
        "torch",
        "torchvision",
        "torchaudio",
        "paralleldomain @ git+https://github.com/parallel-domain/pd-sdk.git",
        "opencv-contrib-python<=3.4.11.45",
        "opencv-python<=3.4.11.45",
        "scikit-image",
    ],
    extras_require={
        "dev": []
    },
    python_requires=">=3.7",
    zip_safe=False,
    entry_points={}
)