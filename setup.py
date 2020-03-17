import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EntropySGD", # Replace with your own username
    version="1.0.0",
    license='MIT',
    author="David Stephane Belemkoabga",
    author_email="bdavidstephane@hotmail.com",
    description="Entropy SGD optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steph1793/EntropySGD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)