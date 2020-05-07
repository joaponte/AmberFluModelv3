
from cc3d import CompuCellSetup
        

from AmberFluModelSteppables import AmberFluModelSteppable

CompuCellSetup.register_steppable(steppable=AmberFluModelSteppable(frequency=1))


CompuCellSetup.run()
