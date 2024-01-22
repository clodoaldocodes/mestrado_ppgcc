from setuptools import setup, find_packages

with open("/home/clodoaldo/mestrado_ppgcc/README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="NeuraNest",
    version="0.0.1",
    author="Clodoaldo Junior",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["/home/clodoaldo/mestrado_ppgcc/functionsToUse.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],
    dependency_links=['']
)