from cc3d.core.PySteppables import *
import numpy as np

plot_StandAlone = False
plot_CellModel = False
overlay_AmbersModel = False
plot_Residuals = False

feedback = True

min_to_mcs = 10.0  # min/mcs
days_to_mcs = min_to_mcs / 1440.0  # day/mcs

'''Smith AP, Moquin DJ, Bernhauerova V, Smith AM. Influenza virus infection model with density dependence 
supports biphasic viral decay. Frontiers in microbiology. 2018 Jul 10;9:1554.'''
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
        if plot_StandAlone:
            self.plot_win = self.add_new_plot_window(title='Amber Smith Model Cells',
                                                 x_axis_title='Days',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False, config_options={'legend': True})
            self.plot_win.add_plot("T", style='Lines', color='red', size=5)
            self.plot_win.add_plot("I1", style='Lines', color='orange', size=5)
            self.plot_win.add_plot("I2", style='Lines', color='green', size=5)
            self.plot_win.add_plot("D", style='Lines', color='purple', size=5)

            self.plot_win2 = self.add_new_plot_window(title='Amber Smith Model Virus',
                                                  x_axis_title='Days',
                                                  y_axis_title='Virus', x_scale_type='linear', y_scale_type='linear',
                                                  grid=False, config_options={'legend': True})
            self.plot_win2.add_plot("V", style='Lines', color='blue', size=5)

    def step(self, mcs):
        self.timestep_sbml()
        if plot_StandAlone:
            self.plot_win.add_data_point("T", mcs * days_to_mcs,self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I1", mcs * days_to_mcs,self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I2", mcs * days_to_mcs,self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("D", mcs * days_to_mcs,self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win2.add_data_point("V", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))


class CellularModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # set initial model parameters
        self.initial_uninfected = len(self.cell_list)  # Scale factor for fraction of cells infected
        self.ExtracellularVirus = self.sbml.ambersmithsimple['V']

        if plot_CellModel:
            self.plot_win3 = self.add_new_plot_window(title='CPM Cells',
                                                    x_axis_title='days',
                                                    y_axis_title='Variables', x_scale_type='linear',
                                                     y_scale_type='linear',
                                                    grid=False,config_options={'legend': True})
            self.plot_win3.add_plot("U", style='Lines', color='red', size=5)
            self.plot_win3.add_plot("I1", style='Lines', color='orange', size=5)
            self.plot_win3.add_plot("I2", style='Lines', color='green', size=5)
            self.plot_win3.add_plot("D", style='Lines', color='purple', size=5)

            if overlay_AmbersModel:
                self.plot_win3.add_plot("AU", style='Dots', color='red', size=5)
                self.plot_win3.add_plot("AI1", style='Dots', color='orange', size=5)
                self.plot_win3.add_plot("AI2", style='Dots', color='green', size=5)
                self.plot_win3.add_plot("AD", style='Dots', color='purple', size=5)

            self.plot_win4 = self.add_new_plot_window(title='CPM Virus',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False,config_options={'legend': True})
            self.plot_win4.add_plot("V", style='Lines', color='blue', size=5)

            if overlay_AmbersModel:
                self.plot_win4.add_plot("AV", style='Dots', color='blue', size=5)

    def step(self, mcs):
        # Transition rule from U to I1
        b = self.sbml.ambersmithsimple['beta'] * days_to_mcs
        if feedback == True:
            V = self.ExtracellularVirus
        else:
            V = self.sbml.ambersmithsimple['V']

        p_UtoI1 = b * V
        for cell in self.cell_list_by_type(self.U):
            if np.random.random() < p_UtoI1:
                cell.type = self.I1

        # Transition rule from I1 to I2
        k = self.sbml.ambersmithsimple['k'] * days_to_mcs
        p_T1oI2 = k
        for cell in self.cell_list_by_type(self.I1):
            if np.random.random() < p_T1oI2:
                cell.type = self.I2

        # Transition rule from I2 to D
        K_delta = self.sbml.ambersmithsimple['K_delta'] / self.sbml.ambersmithsimple['T0'] * self.initial_uninfected
        delta_d = self.sbml.ambersmithsimple['delta_d'] / self.sbml.ambersmithsimple['T0'] * self.initial_uninfected
        I2 = len(self.cell_list_by_type(self.I2))
        p_T2toD = delta_d / (K_delta + I2) * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            if np.random.random() < p_T2toD:
                cell.type = self.DEAD

        # Extracellular Virus
        V = self.ExtracellularVirus
        p = self.sbml.ambersmithsimple['p'] / self.initial_uninfected * self.sbml.ambersmithsimple['T0'] * days_to_mcs
        c = self.sbml.ambersmithsimple['c'] * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            self.ExtracellularVirus += p
        self.ExtracellularVirus -= c * V

        if plot_CellModel:
            self.plot_win3.add_data_point("U", mcs * days_to_mcs, len(self.cell_list_by_type(self.U)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I1", mcs * days_to_mcs,len(self.cell_list_by_type(self.I1)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I2", mcs * days_to_mcs,len(self.cell_list_by_type(self.I2)) / self.initial_uninfected)
            self.plot_win3.add_data_point("D", mcs * days_to_mcs,len(self.cell_list_by_type(self.DEAD)) / self.initial_uninfected)
            self.plot_win4.add_data_point("V", mcs * days_to_mcs, np.log10(self.ExtracellularVirus))

            if overlay_AmbersModel:
                self.plot_win3.add_data_point("AU", mcs * days_to_mcs, self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AI1", mcs * days_to_mcs,self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AI2", mcs * days_to_mcs,self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AD", mcs * days_to_mcs,self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win4.add_data_point("AV", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))


class StatisticsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.initial_uninfected = len(self.cell_list)

        self.cellular_infection =  False
        self.cellular_infection_time = 0.0
        self.Ambersmodel_infection = False
        self.Ambersmodel_infection_time = 0.0
        self.infection_threshold = 0.1

        if plot_Residuals:
            self.plot_win5 = self.add_new_plot_window(title='Residuals',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False,config_options={'legend': True})
            self.plot_win5.add_plot("dU", style='Lines', color='red', size=5)
            self.plot_win5.add_plot("dI1", style='Lines', color='orange', size=5)
            self.plot_win5.add_plot("dI2", style='Lines', color='green', size=5)
            self.plot_win5.add_plot("dD", style='Lines', color='purple', size=5)

    def step(self, mcs):
        if self.cellular_infection == False:
            if len(self.cell_list_by_type(self.I1))/self.initial_uninfected >= self.infection_threshold:
                self.cellular_infection_time = mcs
                self.cellular_infection = True

        if self.Ambersmodel_infection == False:
            if self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'] >= self.infection_threshold:
                self.Ambersmodel_infection_time = mcs
                self.Ambersmodel_infection = True

        print("Cellular Infection = ", self.cellular_infection_time * days_to_mcs)
        print("ODE Infection = ", self.Ambersmodel_infection_time * days_to_mcs)

        dU = (len(self.cell_list_by_type(self.U)) / self.initial_uninfected) - (self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
        dI1 = (len(self.cell_list_by_type(self.I1)) / self.initial_uninfected) - (self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
        dI2 = (len(self.cell_list_by_type(self.I2)) / self.initial_uninfected) - (self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
        dD = (len(self.cell_list_by_type(self.DEAD)) / self.initial_uninfected) - (self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])

        if plot_Residuals:
            self.plot_win5.add_data_point("dU", mcs * days_to_mcs, dU)
            self.plot_win5.add_data_point("dI1", mcs * days_to_mcs, dI1)
            self.plot_win5.add_data_point("dI2", mcs * days_to_mcs, dI2)
            self.plot_win5.add_data_point("dD", mcs * days_to_mcs, dD)

#         # Plot lagged differences between cell populations
#         # Start when both populations are infected
#         # Josh--you will need to save the time series in a set of lists so you can do this
#         # Once both times series have had infection begin
#         # Plot (x(t-starttime)-X(t-cellstarttime)) for each series
#         # Could do same thing to show virus with lags and also RMS deviation with lags
#         #Next step, have virus diffuse and cells infected by viral field rather than the external variable
