import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="leabra",
    version="1.1.15",
    author="emergent",
    author_email="oreilly@ucdavis.edu",
    description="Python interface to emergent neural network simulation system, in Go",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/go-python/gopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
