from cc3d.core.PySteppables import *
import numpy as np

ModelString = '''        
        model ambersmithsimple()
        
        //State Variables and Transitions
        V1: -> T  ; -beta * V * T ;                             // Susceptible Cells
        V2: -> I1 ;  beta * V * T - k * I1 ;                    // Early Infected Cells
        V3: -> I2 ;  k * I1 - delta_d * I2 / (K_delta + I2) ;   // Late Infected Cells
        V4: -> V  ;  p * I2 - c * V ;                           // Extracellular Virus
        V5: -> D  ;  delta_d * I2 / (K_delta + I2) ;            // Cleared Infected Cells (for Bookkeeping)
        
        //Parameters
        beta = 2.4* 10^(-4) ;                                   // Virus Infective
        p = 1.6 ;                                               // Virus Production
        c = 13.0 ;                                              // Virus Clearance
        k = 4.0 ;                                               // Eclipse phase
        delta_d = 1.6 * 10^6 ;                                  // Infected Cell Clearance
        K_delta = 4.5 * 10^5 ;                                  // Half Saturation Constant         
        
        // Initial Conditions ;
        T0 = 1.0*10^7;
        T = T0  ;                                               // Initial Number of Uninfected Cells
        I1 = 75.0 ;                                             // Initial Number of Infected Cells
end'''

min_to_mcs = 10.0  # min/mcs
days_to_mcs = min_to_mcs / 1440.0  # day/mcs


class AmberFluModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Uptading max simulation steps using scaling factor to simulate 10 days
        self.get_xml_element('simulation_steps').cdata = 10.0 / days_to_mcs

        # Adding free floating antimony model
        self.add_free_floating_antimony(model_string=ModelString, model_name='ambersmithsimple',
                                        step_size=days_to_mcs)
        # Changing initial values according to discussions with Amber Smith
        state = {}
        state['I1'] = 0.0
        state['V'] = 75.0
        self.set_sbml_state(model_name='ambersmithsimple', state=state)

        # Initialize Graphic Window for Amber Smith ODE model
        self.plot_win = self.add_new_plot_window(title='Amber Smith Model Cells',
                                                 x_axis_title='Days',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        self.plot_win.add_plot("T", style='Lines', color='red', size=5)
        self.plot_win.add_plot("I1", style='Lines', color='orange', size=5)
        self.plot_win.add_plot("I2", style='Lines', color='green', size=5)
        #self.plot_win.add_plot("D", style='Lines', color='yellow', size=5)

        self.plot_win2 = self.add_new_plot_window(title='Amber Smith Model Virus',
                                                  x_axis_title='Days',
                                                  y_axis_title='Virus', x_scale_type='linear', y_scale_type='linear',
                                                  grid=False)
        self.plot_win2.add_plot("V", style='Lines', color='blue', size=5)

    def step(self, mcs):
        self.timestep_sbml()
        self.plot_win.add_data_point("T", mcs * days_to_mcs,self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
        self.plot_win.add_data_point("I1", mcs * days_to_mcs,self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
        self.plot_win.add_data_point("I2", mcs * days_to_mcs,self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
        #self.plot_win.add_data_point("D", mcs * days_to_mcs,self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
        self.plot_win2.add_data_point("V", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))


class CellularModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # set initial model parameters
        self.initial_uninfected = len(self.cell_list)  # Scale factor for fraction of cells infected
        self.ExtracellularVirus= self.sbml.ambersmithsimple['V']

        self.plot_win3 = self.add_new_plot_window(title='CPM Cells',
                                                  x_axis_title='days',
                                                  y_axis_title='Variables', x_scale_type='linear',
                                                  y_scale_type='linear',
                                                  grid=False)
        self.plot_win3.add_plot("U", style='Lines', color='red', size=5)
        self.plot_win3.add_plot("AU", style='Dots', color='red', size=5)
        self.plot_win3.add_plot("I1", style='Lines', color='orange', size=5)
        self.plot_win3.add_plot("AI1", style='Dots', color='orange', size=5)
        self.plot_win3.add_plot("I2", style='Lines', color='green', size=5)
        self.plot_win3.add_plot("AI2", style='Dots', color='green', size=5)
        # self.plot_win3.add_plot("D", style='Dots', color='blue', size=5)

        self.plot_win4 = self.add_new_plot_window(title='CPM Virus',
                                                  x_axis_title='days',
                                                  y_axis_title='Variables', x_scale_type='linear',
                                                  y_scale_type='linear',
                                                  grid=False)
        self.plot_win4.add_plot("V", style='Lines', color='blue', size=5)
        self.plot_win4.add_plot("AV", style='Dots', color='blue', size=5)

    def step(self, mcs):
        b = self.sbml.ambersmithsimple['beta']
        V = self.sbml.ambersmithsimple['V'] * days_to_mcs

        # Transition rule from U to I1
        p_UtoI1 = b * V
        for cell in self.cell_list_by_type(self.U):
            if np.random.random() < p_UtoI1:
                cell.type = self.I1

        # Transition rule from I1 to I2
        p_T1oI2 = self.sbml.ambersmithsimple['k'] * days_to_mcs
        for cell in self.cell_list_by_type(self.I1):
            if np.random.random() < p_T1oI2:
                cell.type = self.I2

        # Transition rule from I2 to D
        K_delta = self.sbml.ambersmithsimple['K_delta'] / self.sbml.ambersmithsimple['T0'] * self.initial_uninfected
        delta_d = self.sbml.ambersmithsimple['delta_d'] / self.sbml.ambersmithsimple['T0'] * self.initial_uninfected
        I2 = len(self.cell_list_by_type(self.I2))
        p_T2toD = delta_d / (K_delta + I2) * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            print(p_T2toD)
            if np.random.random() < p_T2toD:
                cell.type = self.DEAD

        # Extracellular Virus
        p = self.sbml.ambersmithsimple['p'] * days_to_mcs
        I2 = len(self.cell_list_by_type(self.I2))
        c = self.sbml.ambersmithsimple['c'] * days_to_mcs
        V = self.ExtracellularVirus
        dV = p * I2 - c * V
        self.ExtracellularVirus = V + dV

        self.plot_win3.add_data_point("U", mcs * days_to_mcs, len(self.cell_list_by_type(self.U)) / self.initial_uninfected)
        self.plot_win3.add_data_point("AU", mcs * days_to_mcs, self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
        self.plot_win3.add_data_point("I1", mcs * days_to_mcs,len(self.cell_list_by_type(self.I1)) / self.initial_uninfected)
        self.plot_win3.add_data_point("AI1", mcs * days_to_mcs, self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
        self.plot_win3.add_data_point("I2", mcs * days_to_mcs,len(self.cell_list_by_type(self.I2)) / self.initial_uninfected)
        self.plot_win3.add_data_point("AI2", mcs * days_to_mcs,self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])

        self.plot_win4.add_data_point("V", mcs * days_to_mcs, self.ExtracellularVirus/self.initial_uninfected)
        self.plot_win4.add_data_point("AV", mcs * days_to_mcs, self.sbml.ambersmithsimple['V']/self.sbml.ambersmithsimple['T0'])

#         # Plot lagged differences between cell populations
#         # Start when both populations are infected
#         # Josh--you will need to save the time series in a set of lists so you can do this
#         # Once both times series have had infection begin
#         # Plot (x(t-starttime)-X(t-cellstarttime)) for each series
#
#         # Could do same thing to show virus with lags and also RMS deviation with lags
#
#         #Next step have cells produce virus
#         #Next step, have virus diffuse and cells infected by viral field rather than the external variable
