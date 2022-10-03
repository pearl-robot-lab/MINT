from setuptools import setup, find_packages

setup(
    name='mint',
    package='mint',
    version='0.0.1',
    description="MINT: More Information, Better Transportation: Unsupervised Keypoint Detection and Tracking",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
)