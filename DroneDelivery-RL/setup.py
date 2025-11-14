from setuptools import setup, find_packages

setup(
    name="drone-delivery-rl",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch=1.13.0",
        "numpy=1.21.0",
        "gymnasium=0.29.0",
        "airsim=1.6.0",
        "PyYAML=6.0",
        "pandas=1.4.0",
        "matplotlib=3.5.0",
    ],
    author="DroneDelivery-RL Team",
    description="Indoor Multi-Floor UAV Delivery System using Reinforcement Learning",
    python_requires="=3.8",
)
