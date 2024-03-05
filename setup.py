from setuptools import setup, find_packages

setup(
    name="KLT_filter",
    version="0.0.0",
    author="shionandrew",
    author_email="shionandrew@gmail.com",
    packages=find_packages(where="src", include=['*']),
    package_dir = {
        '':"src"
    },
    include_package_data=True,
    url="https://github.com/shionandrew/KLT_filter",
    install_requires=[
        "numpy >= 1.19.2",
        "scipy >= 1.5.2",
        "matplotlib",
        "h5py",
        "argparse",
        "astropy",
    ],
    python_requires=">=3.6",
)
