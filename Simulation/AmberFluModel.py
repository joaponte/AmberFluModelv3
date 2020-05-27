
from cc3d import CompuCellSetup

from AmberFluModelSteppables import CellularModelSteppable
CompuCellSetup.register_steppable(steppable=CellularModelSteppable(frequency=1))

from AmberFluModelSteppables import Data_OutputSteppable
CompuCellSetup.register_steppable(steppable=Data_OutputSteppable(frequency=1))

CompuCellSetup.run()
