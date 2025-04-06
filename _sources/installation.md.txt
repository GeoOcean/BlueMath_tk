# Installation for Developers

BlueMath is already available via `pip` or `conda`.

To test its capabilities locally, please run the following commands:

1. Clone the repository from GitHub:
```sh
git clone https://github.com/GeoOcean/BlueMath_tk.git
```
2. Move inside the directory to install everything:
```sh
cd BlueMath_tk
```
3. Create a compatible conda environment:
```sh
conda env create -f environment.yml
```
4. Then activate the environment using:
```sh
conda activate bluemath
```
5. Finally, install package in development mode:
```sh
pip install -e .
```