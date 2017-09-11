# Fly
An interactive geophysical flow visualizer for Python.

Geophysicists usually visualize data as static two-dimensional plots on maps.
This approach is well-suited to produce high-quality plots that convey detailed
information to other scientists. However, those plots have two big practical
disadvantages:

- *Exploring* a new data set feels clunky. Matplotlib is not built for speed,
  and is awkward to use interactively.
- Every two-dimensional map projection introduces distortions of some kind - when
  plotting vectors or streamlines, one has to be very careful not to skew the angles
  and magnitudes of the vectors at high latitudes.

Additionally, there is one psychological disadvantage: After many hours of analysis,
one tends to forget the *dynamic* nature of the ocean and atmosphere. Wouldn't
it be nice to see your data move, for once?

**fly** provides an easy way to visualize both scalar and vector fields in real,
interactive 3D. Since it is written in Python, you can do all data processing as
you are used to, and just pass your data as NumPy arrays to fly. And thanks to Glumpy's

## Installation

Most of the heavy lifting is done by Glumpy, but to use fly, you will have to install
some additional Python dependencies. First up, note that fly only supports Python 3,
and only works with Glumpy's recent master (not the release version found on PyPI).

If you have Python 3 and pip installed, you may e.g. run

```
pip3 install -r requirements.txt --user
```

to install all dependencies. Alternatively, you should at least install the following packages:

- NumPy
- SciPy
- Matplotlib
- Cython
- pyopengl
- Glumpy (from GitHub, e.g., via `pip install git+git://github.com/glumpy/glumpy.git`)

## Usage

All code for fly is contained in a single source file, called ``fly.py``. This file contains the Fly class, which handles the visualization for you. All you need to do is to write an interface that reads your data, and pass it to the ``Fly`` class as NumPy arrays:

```python
from fly import Fly

# create Fly instance
fly = Fly(flow_field=velocity)

# apply settings
fly.rotate = True

# do setup
fly.setup()

# run animation
fly.run()
```

Please note that **all input fields must lie on a regular longitude-latitude grid** with bounds (-180,180) and (-90,90). Make sure to interpolate accordingly before passing data to fly.

A (sloppy) interface script for a particular use case (interpreting [Veros](https://github.com/dionhaefner/veros) output) is located in the ``fly_veros.py`` file in this repository. It implements a command line interface
that can e.g. be used like this:

```
python3 fly_veros.py veros_data.nc --flow-variables u v --shading variable --shading-variable temp --depth -10 --time -1
```

All available settings are class attributes of the Fly class, and documented
in ``fly.py``.
