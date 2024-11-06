# Welcome to BlueMath

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

```
├── bluemath_tk
│   ├── __init__.py
│   ├── colors.py
│   ├── datasets.py
│   ├── launchers.py
│   ├── README.md
│   ├── sketch_tk.png
│   ├── core
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── models.py
│   │   ├── prueba.py
│   │   └── utils.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── bathy.py
│   │   ├── downloaders.py
│   │   ├── extraction.py
│   │   └── visualization.py
│   ├── datamining
│   │   ├── __init__.py
│   │   ├── kma.py
│   │   ├── lhs.py
│   │   ├── mda.py
│   │   ├── pca.py
│   │   ├── prueba.py
│   │   └── som.py
│   ├── distributions
│   │   ├── __init__.py
│   │   ├── copula.py
│   │   ├── gev.py
│   │   ├── gpd.py
│   │   ├── poisson.py
│   │   └── pot.py
│   ├── interp
│   │   ├── __init__.py
│   │   ├── analogs.py
│   │   ├── gps.py
│   │   └── rbf.py
│   ├── predictor
│   │   ├── __init__.py
│   │   ├── awt.py
│   │   ├── dwt.py
│   │   ├── indices.py
│   │   ├── itca.py
│   │   └── iwt.py
│   ├── risk
│   │   ├── __init__.py
│   │   ├── damage.py
│   │   ├── pcrafi.py
│   │   └── riskscapetools.py
│   ├── tc
│   │   ├── __init__.py
│   │   ├── graffitiwaves.py
│   │   ├── parameterization.py
│   │   ├── qtcrain.py
│   │   ├── skytcwaves.py
│   │   ├── tracks.py
│   │   └── vortex.py
│   ├── tide
│   │   ├── __init__.py
│   │   ├── harmonic.py
│   │   ├── ttide.py
│   │   └── utide.py
│   ├── topo-bathy
│   │   ├── __init__.py
│   │   └── profiles.py
│   ├── waves
│   │   ├── __init__.py
│   │   ├── climate.py
│   │   ├── estela.py
│   │   ├── greenswell.py
│   │   ├── partitioning.py
│   │   ├── snakes.py
│   │   └── superpoint.py
│   └── wrappers
│       ├── __init__.py
│       ├── cgwave.py
│       ├── delft3d.py
│       ├── lisflood.py
│       ├── schism.py
│       ├── swan.py
│       ├── swash.py
│       └── xbeach.py
├── tests
│   └── datamining
│       └── test_mda.py
├── demos
│   └── datamining
│       ├── kma.py
│       ├── mda.py
│       └── README.md
├── docs
│   ├── api.md
│   └── index.md
├── logs
│   └── MDA_2024-11-05.log
├── LICENSE.txt
├── mkdocs.yml
├── README.md
├── requirements.txt
└── setup.py
```