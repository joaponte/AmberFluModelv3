
from cc3d import CompuCellSetup
        
from AmberFluModelSteppables import AmberFluModelSteppable
CompuCellSetup.register_steppable(steppable=AmberFluModelSteppable(frequency=1))

from AmberFluModelSteppables import CellularModelSteppable
CompuCellSetup.register_steppable(steppable=CellularModelSteppable(frequency=1))

from AmberFluModelSteppables import StatisticsSteppable
CompuCellSetup.register_steppable(steppable=StatisticsSteppable(frequency=1))

CompuCellSetup.run()
