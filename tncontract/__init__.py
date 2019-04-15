"""
tncontract
==========

A simple tensor-network library.

Available subpackages
---------------------
onedim
    Special purpose classes, methods and function for one dimensional
    tensor networs

twodim
    Special purpose classes, methods and function for two dimensional
    tensor networs
"""

from tncontract.version import __version__
import tncontract.eksp as eksp
from tncontract.tensor_core import *
from tncontract.tensor_instant import *
from tncontract.label import *
import tncontract.matrices as matrices
import tncontract.eiglib as eiglib
import tncontract.onedim as onedim
import tncontract.varopt as varopt
