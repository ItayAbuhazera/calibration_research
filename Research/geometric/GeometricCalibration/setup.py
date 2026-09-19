from setuptools import setup, find_packages

setup(
    name="calibrators",
    version="0.1.0",
    author="Itay Abuhazera",
    author_email="itay0011@gmail.com",
    description="A Python library for model calibration with a focus on geometric calibration methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ItayAbuhazera/Calibrato",  # Replace with your GitHub repository link
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "matplotlib",
        # Add more dependencies here if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
