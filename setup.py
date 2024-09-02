from setuptools import setup, find_packages

install_requires = []

setup(
    name="adaptigraph",
    version="1.0.0",
    # install_requires=read_requirements(),
    # packages=["adaptigraph"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
