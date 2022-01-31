from setuptools import find_packages, setup

setup(
    name="rllib_policies",
    description="Neural Nets for RLlib",
    packages=find_packages("src"),
    python_requires=">=3.7",
    package_dir={"": "src"},
    install_requires=["numpy<=1.19", "ray[rllib]>0.8.0", "torch>=1.4.0"],
)
