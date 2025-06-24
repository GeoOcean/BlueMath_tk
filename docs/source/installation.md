# Installation for Developers

BlueMath is already available via `pip`.

To test its capabilities locally, please run the following commands:

1. Clone the repository from GitHub:

```sh
git clone https://github.com/GeoOcean/BlueMath_tk.git
```

2. Move inside the directory to install everything:

```sh
cd BlueMath_tk
```

3. Create a compatible environment with *Python >= 3.11* (example with conda):

```sh
conda create -n bluemath-dev python=3.11
```

4. Then activate the environment using:

```sh
conda activate bluemath-dev
```

5. Change `pyproject.toml` name to avoid errors:

```
mv pyproject.toml pyproject.toml.bak
```

6. Finally, install package in development mode:

```sh
pip install -e .
```