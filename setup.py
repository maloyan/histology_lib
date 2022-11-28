from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="histology_lib",
    packages=find_packages(),
    version="0.0.1",
    description="histology classification",
    author="Narek Maloyan",
    license="MIT",
)