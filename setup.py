from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="urbancode",
    version="0.1.1",
    author="Sijie Yang",
    author_email="sijiey@u.nus.edu",
    description="A package for universal urban analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sijie-Yang/urbancode",
    project_urls={
        "Bug Tracker": "https://github.com/Sijie-Yang/urbancode/issues",
        "Changelog": "https://github.com/Sijie-Yang/urbancode/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.5",
        "osmnx>=1.1.1",
        "momepy>=0.5.3",
        "geopandas>=0.9.0",
        "matplotlib>=3.3.4",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.0.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.5.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.1",
        "ipywidgets>=8.0.0",
        "jupyter>=1.0.0",
        "urllib3>=2.0.0",
        "requests>=2.0.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'black>=21.0',
            'isort>=5.0'
        ]
    }
)