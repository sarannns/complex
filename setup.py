import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="complexnetwork",
    version="0.1.2",
    author="Saranraj Nambusubramaniyan",
    author_email="saran_nns@hotmail.com",
    description="Complex networks and Self-Organized Criticality",
    license="OSI Approved :: MIT License",
    keywords="""Brain-Inspired Computing, Complex Networks, Criticality, Artificial Neural Networks,Neuro Informatics, 
                  Spiking Cortical Networks, Neural Connectomics,Neuroscience, Artificial General Intelligence, Neural Information Processing""",
    url="https://github.com/Saran-nns/self_organized_criticality",
    packages=["complexnetwork"],
    data_files=["LICENSE"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    install_requires=["numpy", "scipy", "seaborn"],
    zip_safe=False,
)

