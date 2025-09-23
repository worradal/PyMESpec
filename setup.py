from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pymespec",
    version="1.2.1",
    author="Alfred Worrad", 
    author_email="worrada@udel.com", 
    description="PyMESpec is designed to be an encompassing software package for the analysis of transient experimental spectroscopic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worradal/PyMESpec", # Need to update this after new repo
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pymespec=src.main:main",  # Adjust if you have a main function
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
