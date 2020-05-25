from cc3d.core.PySteppables import *
import numpy as np
import os

plot_ODEStandAlone = False
## Plot Coinfection ODE
#  1 plot virus A
#  2 plot virus B
#  3 plot both viruses
plot_ConinfectionODEtandAlone = False
overlay_AmbersModelandCoinfection = False
plot_CellModel = True
overlay_AmbersModel = False
plot_Residuals = False
Data_writeout = False

## How to determine V
# -1 pulls from the scalar virus from the ODE original model (no feedback in the cellular model)
#  0 pulls from the scalar virus from the cellular model (feedback in the cellular model but no field)
#  1 pulls from the virus field
how_to_determine_V = 1

min_to_mcs = 10.0  # min/mcs
days_to_mcs = min_to_mcs / 1440.0  # day/mcs
days_to_simulate = 4.0 #10 in the original model

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

Coinfection_ModelString = '''
        model coinfection()
         //State Variables and Transitions for Virus A
        V1: -> T   ; -beta * VA * T ;                               // Susceptible Cells
        V2: -> I1A ;  beta * VA * T - k * I1A ;                     // Early Infected Cells
        V3: -> I2A ;  k * I1A - delta_d * I2A / (K_delta + I2A) ;   // Late Infected Cells
        V4: -> VA  ;  p * I2A - c * VA;                             // Extracellular Virus A
        V5: -> DA  ;  delta_d * I2A / (K_delta + I2A) ;             // Dead Cells

         //State Variables and Transitions for Virus B
        V6: -> T   ; -beta * VB * T ;                               // Susceptible Cells
        V7: -> I1B ;  beta * VB * T - k * I1B ;                     // Early Infected Cells
        V8: -> I2B ;  k * I1B - delta_d * I2B / (K_delta + I2B) ;   // Late Infected Cells
        V9: -> VB  ;  p * I2B - c * VB;                             // Extracellular Virus B
        V10: -> DB  ;  delta_d * I2B / (K_delta + I2B) ;             // Dead Cells

        //Parameters
        beta = 2.4* 10^(-4) ;                                       // Virus Infective
        p = 1.6 ;                                                   // Virus Production
        c = 13.0 ;                                                  // Virus Clearance
        k = 4.0 ;                                                   // Eclipse phase
        delta_d = 1.6 * 10^6 ;                                      // Infected Cell Clearance
        K_delta = 4.5 * 10^5 ;                                      // Half Saturation Constant   

        // Initial Conditions ;
        T0 = 1.0*10^7;
        T = T0  ;                                                   // Initial Number of Uninfected Cells
        I1 = 75.0 ;                                                 // Initial Number of Infected Cells
end'''

class AmberFluModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Uptading max simulation steps using scaling factor to simulate 10 days
        self.get_xml_element('simulation_steps').cdata = days_to_simulate / days_to_mcs

        # Adding free floating single antimony model
        self.add_free_floating_antimony(model_string=ModelString, model_name='ambersmithsimple',
                                        step_size=days_to_mcs)
        # Changing initial values according to discussions with Amber Smith
        state = {}
        state['I1'] = 0.0
        state['V'] = 75.0
        self.set_sbml_state(model_name='ambersmithsimple', state=state)

        # Adding free floating coinfection antimony model
        self.add_free_floating_antimony(model_string=Coinfection_ModelString, model_name='coinfection',
                                        step_size=days_to_mcs)
        # Changing initial values according to discussions with Amber Smith
        state = {}
        state['I1'] = 0.0
        state['VA'] = 75.0
        state['VB'] = 75.0
        self.set_sbml_state(model_name='coinfection', state=state)

        # Initialize Graphic Window for Amber Smith ODE model
        if plot_ODEStandAlone:
            self.plot_win = self.add_new_plot_window(title='Amber Smith Model Cells',
                                                     x_axis_title='Days',
                                                     y_axis_title='Variables', x_scale_type='linear',
                                                     y_scale_type='linear',
                                                     grid=False, config_options={'legend': True})
            self.plot_win.add_plot("T", style='Lines', color='blue', size=5)
            self.plot_win.add_plot("I1", style='Lines', color='yellow', size=5)
            self.plot_win.add_plot("I2", style='Lines', color='red', size=5)
            self.plot_win.add_plot("D", style='Lines', color='purple', size=5)

            self.plot_win2 = self.add_new_plot_window(title='Amber Smith Model Virus',
                                                      x_axis_title='Days',
                                                      y_axis_title='Virus', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})
            self.plot_win2.add_plot("V", style='Lines', color='blue', size=5)

            # Initialize Graphic Window for Coninfection ODE model
        if plot_ConinfectionODEtandAlone:
            self.plot_win6 = self.add_new_plot_window(title='Coinfection ODE Model Cells',
                                                     x_axis_title='Days',
                                                     y_axis_title='Variables', x_scale_type='linear',
                                                     y_scale_type='linear',
                                                     grid=False, config_options={'legend': True})

            self.plot_win7 = self.add_new_plot_window(title='Coinfection ODE Model Virus',
                                                      x_axis_title='Days',
                                                      y_axis_title='Virus', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})

            self.plot_win6.add_plot("T", style='Lines', color='blue', size=5)
            if plot_ConinfectionODEtandAlone == 1 or plot_ConinfectionODEtandAlone == 3:
                self.plot_win6.add_plot("I1A", style='Lines', color='yellow', size=5)
                self.plot_win6.add_plot("I2A", style='Lines', color='red', size=5)
                self.plot_win6.add_plot("DA", style='Lines', color='purple', size=5)
                self.plot_win7.add_plot("VA", style='Lines', color='blue', size=5)
            if plot_ConinfectionODEtandAlone == 2 or plot_ConinfectionODEtandAlone == 3:
                self.plot_win6.add_plot("I1B", style='Lines', color='green', size=5)
                self.plot_win6.add_plot("I2B", style='Lines', color='magenta', size=5)
                self.plot_win6.add_plot("DB", style='Lines', color='brown', size=5)
                self.plot_win7.add_plot("VB", style='Lines', color='cyan', size=5)
            if overlay_AmbersModelandCoinfection:
                self.plot_win6.add_plot("TAmber", style='Dots', color='blue', size=5)
                self.plot_win6.add_plot("I1Amber", style='Dots', color='yellow', size=5)
                self.plot_win6.add_plot("I2Amber", style='Dots', color='red', size=5)
                self.plot_win6.add_plot("DAmber", style='Dots', color='purple', size=5)
                self.plot_win7.add_plot("VAmber", style='Dots', color='blue', size=5)

    def step(self, mcs):
        self.timestep_sbml()
        if plot_ODEStandAlone:
            self.plot_win.add_data_point("T", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I1", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I2", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("D", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win2.add_data_point("V", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))

        if plot_ConinfectionODEtandAlone:
            self.plot_win6.add_data_point("T", mcs * days_to_mcs,
                                         self.sbml.coinfection['T'] / self.sbml.coinfection['T0'])

            if plot_ConinfectionODEtandAlone == 1 or plot_ConinfectionODEtandAlone == 3:
                self.plot_win6.add_data_point("I1A", mcs * days_to_mcs,
                                             self.sbml.coinfection['I1A'] / self.sbml.coinfection['T0'])
                self.plot_win6.add_data_point("I2A", mcs * days_to_mcs,
                                             self.sbml.coinfection['I2A'] / self.sbml.coinfection['T0'])
                self.plot_win6.add_data_point("DA", mcs * days_to_mcs,
                                             self.sbml.coinfection['DA'] / self.sbml.coinfection['T0'])
                self.plot_win7.add_data_point("VA", mcs * days_to_mcs, np.log10(self.sbml.coinfection['VA']))

            if plot_ConinfectionODEtandAlone == 2 or plot_ConinfectionODEtandAlone == 3:
                self.plot_win6.add_data_point("I1B", mcs * days_to_mcs,
                                             self.sbml.coinfection['I1B'] / self.sbml.coinfection['T0'])
                self.plot_win6.add_data_point("I2B", mcs * days_to_mcs,
                                             self.sbml.coinfection['I2B'] / self.sbml.coinfection['T0'])
                self.plot_win6.add_data_point("DB", mcs * days_to_mcs,
                                             self.sbml.coinfection['DB'] / self.sbml.coinfection['T0'])
                self.plot_win7.add_data_point("VB", mcs * days_to_mcs, np.log10(self.sbml.coinfection['VB']))

            if overlay_AmbersModelandCoinfection:
                self.plot_win6.add_data_point("TAmber", mcs * days_to_mcs,
                                             self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win6.add_data_point("I1Amber", mcs * days_to_mcs,
                                             self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win6.add_data_point("I2Amber", mcs * days_to_mcs,
                                             self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win6.add_data_point("DAmber", mcs * days_to_mcs,
                                             self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win7.add_data_point("VAmber", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))


class CellularModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # set initial model parameters
        self.initial_uninfected = len(self.cell_list_by_type(self.U))
        self.ExtracellularVirus = self.sbml.coinfection['VA']
        self.get_xml_element('virus_decay').cdata = self.sbml.ambersmithsimple['c'] * days_to_mcs
        self.ExtracellularVirusB = self.sbml.coinfection['VB']
        self.get_xml_element('virusB_decay').cdata = self.sbml.ambersmithsimple['c'] * days_to_mcs

        if plot_CellModel:
            self.plot_win3 = self.add_new_plot_window(title='CPM Cells',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})
            self.plot_win3.add_plot("U", style='Lines', color='blue', size=5)
            self.plot_win3.add_plot("I1A", style='Lines', color='yellow', size=5)
            self.plot_win3.add_plot("I2A", style='Lines', color='red', size=5)
            self.plot_win3.add_plot("DA", style='Lines', color='purple', size=5)
            self.plot_win3.add_plot("I1B", style='Lines', color='green', size=5)
            self.plot_win3.add_plot("I2B", style='Lines', color='magenta', size=5)
            self.plot_win3.add_plot("DB", style='Lines', color='brown', size=5)

            if overlay_AmbersModel:
                self.plot_win3.add_plot("AU", style='Dots', color='blue', size=5)
                self.plot_win3.add_plot("AI1", style='Dots', color='yellow', size=5)
                self.plot_win3.add_plot("AI2", style='Dots', color='red', size=5)
                self.plot_win3.add_plot("AD", style='Dots', color='purple', size=5)

            self.plot_win4 = self.add_new_plot_window(title='CPM Virus',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})
            self.plot_win4.add_plot("VA", style='Lines', color='blue', size=5)
            self.plot_win4.add_plot("VB", style='Lines', color='cyan', size=5)

            if overlay_AmbersModel:
                self.plot_win4.add_plot("AV", style='Dots', color='blue', size=5)

    def step(self, mcs):
        # Transition rule from U to I1
        secretor = self.get_field_secretor("Virus")
        secretorB = self.get_field_secretor("VirusB")
        for cell in self.cell_list_by_type(self.U):
            # Determine V from scalar virus from the ODE
            if how_to_determine_V == -1:
                b = self.sbml.coinfection['beta'] * self.sbml.coinfection['T0'] * days_to_mcs
                VA = self.sbml.coinfection['VA'] / self.sbml.coinfection['T0']
                VB = self.sbml.coinfection['VB'] / self.sbml.coinfection['T0']

            # Determine V from scalar virus from the cellular model
            if how_to_determine_V == 0:
                b = self.sbml.coinfection['beta'] * self.initial_uninfected * days_to_mcs
                VA = self.ExtracellularVirus / self.initial_uninfected
                VB = self.ExtracellularVirusB / self.initial_uninfected

            # Determine V from the virus field
            if how_to_determine_V == 1:
                b = self.sbml.coinfection['beta'] * self.initial_uninfected * days_to_mcs
                uptake_probability = 0.0000001

                uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                VA = abs(uptake.tot_amount) / uptake_probability
                secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

                uptakeB = secretorB.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                VB = abs(uptakeB.tot_amount) / uptake_probability
                secretorB.secreteInsideCellTotalCount(cell, abs(uptakeB.tot_amount) / cell.volume)

            # Calculate the probability of infection of individual cells based on the amount of virus PER cell
            # Transition rule from T to I2
            p_UtoI1A = b * VA
            if np.random.random() < p_UtoI1A:
                cell.type = self.I1

            # Transition rule from T to I2B
            p_UtoI1A = b * VB
            if np.random.random() < p_UtoI1A:
                cell.type = self.I1B

        # Transition rule from I1 to I2
        k = self.sbml.coinfection['k'] * days_to_mcs
        p_T1oI2 = k
        for cell in self.cell_list_by_type(self.I1):
            if np.random.random() < p_T1oI2:
                cell.type = self.I2

        # Transition rule from I1B to I2B
        k = self.sbml.coinfection['k'] * days_to_mcs
        p_T1BoI2B = k
        for cell in self.cell_list_by_type(self.I1B):
            if np.random.random() < p_T1BoI2B:
                cell.type = self.I2B

        # Transition rule from I2 to D
        K_delta = self.sbml.coinfection['K_delta'] / self.sbml.coinfection['T0'] * self.initial_uninfected
        delta_d = self.sbml.coinfection['delta_d'] / self.sbml.coinfection['T0'] * self.initial_uninfected
        I2 = len(self.cell_list_by_type(self.I2))
        p_T2toD = delta_d / (K_delta + I2) * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            if np.random.random() < p_T2toD:
                cell.type = self.DEAD

        # Transition rule from I2B to DB
        K_delta = self.sbml.coinfection['K_delta'] / self.sbml.coinfection['T0'] * self.initial_uninfected
        delta_d = self.sbml.coinfection['delta_d'] / self.sbml.coinfection['T0'] * self.initial_uninfected
        I2B = len(self.cell_list_by_type(self.I2B))
        p_T2BtoDB = delta_d / (K_delta + I2B) * days_to_mcs
        for cell in self.cell_list_by_type(self.I2B):
            if np.random.random() < p_T2BtoDB:
                cell.type = self.DEADB

        # Production of extracellular virus A
        secretor = self.get_field_secretor("Virus")
        V = self.ExtracellularVirus
        p = self.sbml.coinfection['p'] / self.initial_uninfected * self.sbml.coinfection['T0'] * days_to_mcs
        c = self.sbml.coinfection['c'] * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            release = secretor.secreteInsideCellTotalCount(cell, p / cell.volume)
            self.ExtracellularVirus += release.tot_amount
        self.ExtracellularVirus -= c * V

        # Production of extracellular virus A
        secretor = self.get_field_secretor("VirusB")
        VB = self.ExtracellularVirusB
        p = self.sbml.coinfection['p'] / self.initial_uninfected * self.sbml.coinfection['T0'] * days_to_mcs
        c = self.sbml.coinfection['c'] * days_to_mcs
        for cell in self.cell_list_by_type(self.I2B):
            release = secretor.secreteInsideCellTotalCount(cell, p / cell.volume)
            self.ExtracellularVirusB += release.tot_amount
        self.ExtracellularVirusB -= c * VB

        # Measure amount of extracellular virus field
        secretor = self.get_field_secretor("Virus")
        self.ExtracellularVirus_Field = 0
        for cell in self.cell_list:
            uptake_probability = 0.0000001
            uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
            V = abs(uptake.tot_amount) / uptake_probability
            self.ExtracellularVirus_Field += V
            secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

        # Measure amount of extracellular virus field
        secretor = self.get_field_secretor("VirusB")
        self.ExtracellularVirus_FieldB = 0
        for cell in self.cell_list:
            uptake_probability = 0.0000001
            uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
            V = abs(uptake.tot_amount) / uptake_probability
            self.ExtracellularVirus_FieldB += V
            secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

        if plot_CellModel:
            self.plot_win3.add_data_point("U", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.U)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I1A", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I1)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I2A", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I2)) / self.initial_uninfected)
            self.plot_win3.add_data_point("DA", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.DEAD)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I1B", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I1B)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I2B", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I2B)) / self.initial_uninfected)
            self.plot_win3.add_data_point("DB", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.DEADB)) / self.initial_uninfected)
            if how_to_determine_V == 1:
                self.plot_win4.add_data_point("VA", mcs * days_to_mcs, np.log10(self.ExtracellularVirus_Field))
                self.plot_win4.add_data_point("VB", mcs * days_to_mcs, np.log10(self.ExtracellularVirus_FieldB))
            else:
                self.plot_win4.add_data_point("VA", mcs * days_to_mcs, np.log10(self.ExtracellularVirus))
                self.plot_win4.add_data_point("VB", mcs * days_to_mcs, np.log10(self.ExtracellularVirusB))

            if overlay_AmbersModel:
                self.plot_win3.add_data_point("AU", mcs * days_to_mcs,
                                              self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AI1", mcs * days_to_mcs,
                                              self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AI2", mcs * days_to_mcs,
                                              self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win3.add_data_point("AD", mcs * days_to_mcs,
                                              self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
                self.plot_win4.add_data_point("AV", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))

class StatisticsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.initial_uninfected = len(self.cell_list)

        self.cellular_infection = False
        self.cellular_infection_time = 0.0
        self.Ambersmodel_infection = False
        self.Ambersmodel_infection_time = 0.0
        self.infection_threshold = 0.1

        if plot_Residuals:
            self.plot_win5 = self.add_new_plot_window(title='Residuals',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})
            self.plot_win5.add_plot("dU", style='Lines', color='blue', size=5)
            self.plot_win5.add_plot("dI1", style='Lines', color='yellow', size=5)
            self.plot_win5.add_plot("dI2", style='Lines', color='red', size=5)
            self.plot_win5.add_plot("dD", style='Lines', color='purple', size=5)

    def step(self, mcs):
        if self.cellular_infection == False:
            if len(self.cell_list_by_type(self.I1)) / self.initial_uninfected >= self.infection_threshold:
                self.cellular_infection_time = mcs
                self.cellular_infection = True

        if self.Ambersmodel_infection == False:
            if self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'] >= self.infection_threshold:
                self.Ambersmodel_infection_time = mcs
                self.Ambersmodel_infection = True

        # print("Cellular Infection = ", self.cellular_infection_time * days_to_mcs)
        # print("ODE Infection = ", self.Ambersmodel_infection_time * days_to_mcs)

        dU = (len(self.cell_list_by_type(self.U)) / self.initial_uninfected) - (
                self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
        dI1 = (len(self.cell_list_by_type(self.I1)) / self.initial_uninfected) - (
                self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
        dI2 = (len(self.cell_list_by_type(self.I2)) / self.initial_uninfected) - (
                self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
        dD = (len(self.cell_list_by_type(self.DEAD)) / self.initial_uninfected) - (
                self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])

        if plot_Residuals:
            self.plot_win5.add_data_point("dU", mcs * days_to_mcs, dU)
            self.plot_win5.add_data_point("dI1", mcs * days_to_mcs, dI1)
            self.plot_win5.add_data_point("dI2", mcs * days_to_mcs, dI2)
            self.plot_win5.add_data_point("dD", mcs * days_to_mcs, dD)


class Data_OutputSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        if Data_writeout:
            folder_path = '/Users/Josua/Downloads/AmberFluModelv3/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_name = 'AmberFluModel.txt'
            self.output = open(folder_path + file_name, 'w')
            self.output.write(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % ('Time','AT', 'AI1', 'AI2', 'AD', 'AV', 'U', 'I1', 'I2', 'D', 'V'))
            self.output.flush()
        else:
            pass

    def step(self, mcs):
        if Data_writeout:
            # Record variables from ODE model
            AT = self.sbml.ambersmithsimple['T']
            AI1 = self.sbml.ambersmithsimple['I1']
            AI2 = self.sbml.ambersmithsimple['I2']
            AD = self.sbml.ambersmithsimple['D']
            AV = self.sbml.ambersmithsimple['V']

            # Record variables from Cellularized Model
            d = mcs * days_to_mcs
            U = len(self.cell_list_by_type(self.U))
            I1 = len(self.cell_list_by_type(self.I1))
            I2 = len(self.cell_list_by_type(self.I2))
            D = len(self.cell_list_by_type(self.DEAD))

            self.Virus_Field = 0
            secretor = self.get_field_secretor("Virus")
            for cell in self.cell_list:
                uptake_probability = 0.0000001
                uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                V = abs(uptake.tot_amount) / uptake_probability
                self.Virus_Field += V
                secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

            self.output.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
            d, AT, AI1, AI2, AD, AV, U, I1, I2, D, self.Virus_Field))
            self.output.flush()

    def finish(self):
        if Data_writeout:
            self.output.close()
#         # Plot lagged differences between cell populations
#         # Start when both populations are infected
#         # Josh--you will need to save the time series in a set of lists so you can do this
#         # Once both times series have had infection begin
#         # Plot (x(t-starttime)-X(t-cellstarttime)) for each series
#         # Could do same thing to show virus with lags and also RMS deviation with lags
