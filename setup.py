from setuptools import setup, find_packages

setup(
    name="gitstarproject",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "pathlib",
        "sklearn",
        "requests",
        "matplotlib",
        "arrow",
        "pyodbc",
        "seaborn",
        "ipython",
        "ipdb",
    ],
)
