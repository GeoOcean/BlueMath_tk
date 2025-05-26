#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import netCDF4 as nc

# tk
from ..io.matlab import ReadMatfile


def description(p):
    'returns description of the file'
    print('getting info... {0}'.format(p))

    txt = '\n\n\n-->{0}'.format(p)
    if op.isdir(p):
        txt += '\n*** FOLDER ***\n\nfiles:\n'
        txt += ', '.join(os.listdir(p))
        txt += '\n'

    elif p.endswith('.nc'):
        txt += '\n*** NETCDF File ***\n\n'
        txt += str(nc.Dataset(p))

    elif p.endswith('.mat'):
        txt += '\n*** MATLAB File ***\n\nvariables:\n'
        try:
            txt += ', '.join(ReadMatfile(p).keys())
        except:
            txt +='\n couldn\'t read file.'
        txt += '\n'

    return txt
