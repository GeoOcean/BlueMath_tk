from setuptools import setup, find_packages

setup(
    name="BlueMath",
    version="1.0.0",
    author="GeoOcean Group",
    author_email="@GeoOcean",
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GeoOcean/BlueMath",
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,
    install_requires=[  # List your package dependencies here
        "numpy",  # Example dependency
        "pandas",
    ],
    classifiers=["Programming Language :: Python :: 3.6"],
    python_requires=">=3.6",  # Specify the Python version required
)
