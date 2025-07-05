"""Setup configuration for GeoFeatureKit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="geofeaturekit",
    version="0.1.1",
    description="A Python library for extracting and analyzing urban features from OpenStreetMap data",
    author="Alexander Li",
    author_email="lihangalex@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lihangalex/geofeaturekit",
    project_urls={
        "Bug Tracker": "https://github.com/lihangalex/geofeaturekit/issues",
        "Documentation": "https://github.com/lihangalex/geofeaturekit#readme",
        "Source Code": "https://github.com/lihangalex/geofeaturekit",
    },
    packages=find_packages(include=["geofeaturekit", "geofeaturekit.*"]),
    package_data={
        "geofeaturekit": ["py.typed"],
    },
    python_requires=">=3.9",
    install_requires=requirements,
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
        "Topic :: Scientific/Engineering :: GIS",
    ],
) 