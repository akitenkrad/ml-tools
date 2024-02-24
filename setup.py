import sys

import setuptools

with open("README.md", mode="r") as f:
    long_description = f.read()

version_range_max = max(sys.version_info[1], 10) + 1
python_min_version = (3, 11, 0)

setuptools.setup(
    name="ml_tools",
    version="0.0.1",
    author="akitenkrad",
    author_email="akitenkrad@gmail.com",
    packages=setuptools.find_packages(),
    package_data={
        "ml_tools": [
            "config/*.yml",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
    + ["Programming Language :: Python :: 3.{}".format(i) for i in range(python_min_version[1], version_range_max)],
    long_description=long_description,
    install_requires=[
        "attrdict @ git+https://github.com/akitenkrad/attrdict",
        "beautifulsoup4",
        "click",
        "colorama",
        "gensim",
        "ipython",
        "ipywidgets",
        "mlflow",
        "nltk",
        "numpy",
        "pandas",
        "plotly",
        "progressbar",
        "py-cpuinfo",
        "python-dateutil",
        "python-dotenv",
        "pyunpack",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "transformers",
        "toml",
        "torch",
        "torchinfo",
        "tqdm",
    ],
)
