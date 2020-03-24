import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "medreaders",
    version = "1.0.1",
    author = "Olga Senyukova",
    author_email = "olga.senyukova@graphics.cs.msu.ru",
    description = "Readers for medical imaging datasets",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ol-sen/medreaders",
    packages = setuptools.find_packages(),
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix", 
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires = '>=3.6',
    install_requires = [
        "imageio==2.8.0",
        "matplotlib==3.1.3",
        "nibabel==3.0.1",
        "numpy==1.18.1",
        "scikit-image==0.16.2",
    ],
)
