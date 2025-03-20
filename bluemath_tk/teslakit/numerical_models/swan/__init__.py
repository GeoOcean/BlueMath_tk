"""
Module attrs
"""

__version__     = '0.1.0'
__author__      = 'GeoOcean(UC)'
__contact__     = 'ripolln@unican.es'
__url__         = 'https://gitlab.com/geoocean/bluemath/mw_deltares/'
__description__ = 'SWAN numerical model python wrap'
__keywords__    = 'SWAN waves numerical'

import os.path as op
import shutil

from . import geo
from . import io
from . import storms
from . import plots
from . import wrap



def set_swan_binary_file(bin_file):
    'copy swan bin_file to wrap bin location'

    # swan bin executable
    p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')
    p_bin = op.abspath(op.join(p_res, 'swan_bin', 'swan_ser.exe'))

    # copy file
    shutil.copyfile(bin_file, p_bin)

    print('bin file copied to {0}'.format(p_bin))

