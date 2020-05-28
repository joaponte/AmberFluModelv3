from cc3d.core.PySteppables import *
import numpy as np
import os

plot_StandAlone = True
plot_CellModel = True
overlay_AmbersModel = True
plot_Residuals = False
Data_writeout = True

## How to determine V
# -1 pulls from the scalar virus from the ODE original model (no feedback in the cellular model)
#  0 pulls from the scalar virus from the cellular model (feedback in the cellular model but no field)
#  1 pulls from the virus field
how_to_determine_V = 1

min_to_mcs = 10.0  # min/mcs
days_to_mcs = min_to_mcs / 1440.0  # day/mcs
days_to_simulate = 4.0 #10 in the original model

model_string = '''
    //Equations
    E2: -> IFN ; P*(k11*RIGI*V + k12*V^n/(k13+V^n) + k14*IRF7P) ;
    E3A: IFN -> IFNe; k21*IFN ;
    E3B: IFNe -> ; t2*IFNe
    E4A: -> STATP ; k31*P*IFNe/(k32+k33*IFNe) 
    E4B: STATP -> ; t3*STATP
    E5A: -> IRF7 ; P*(k41*STATP + k42*IRF7P)
    E5B: IRF7 -> ; t4*IRF7
    E6A: -> IRF7P ;  k51*P*IRF7
    E7B: IRF7P -> ; t5*IRF7P
    E7A: P -> ; -k61*P*V
    E8A: -> V ; k71*P*V/(1+k72*IFNe)
    E8B: V -> ; k73*V

    //Parameters
    k11 = 0.0 ; //Validation data dependent ; zero for most cases
    k12 = 9.746
    k13 = 12.511
    k14 = 13.562
    k21 = 10.385
    t2 = 3.481
    k31 = 45.922
    k32 = 5.464
    k33 = 0.068
    t3 = 0.3
    k41 = 0.115
    k42 = 1.053
    t4 = 0.3
    k51 = 0.202
    t5 = 0.3
    k61 = 0.635
    k71 = 1.537
    k72 = 47.883
    k73 = 0.197
    n = 3.0

    //Initial Conditions
    P = 1.0
    RIGI = 1.0
    IRF7 = 0.72205
    V = 6.9e-8
'''


class AmberFluModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Uptading max simulation steps using scaling factor to simulate 10 days
        self.get_xml_element('simulation_steps').cdata = days_to_simulate / days_to_mcs

        # Adding free floating antimony model
        self.add_free_floating_antimony(model_string=model_string, model_name='ambersmithsimple',
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

    def step(self, mcs):
        self.timestep_sbml()
        if plot_StandAlone:
            self.plot_win.add_data_point("T", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['T'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I1", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['I1'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("I2", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['I2'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win.add_data_point("D", mcs * days_to_mcs,
                                         self.sbml.ambersmithsimple['D'] / self.sbml.ambersmithsimple['T0'])
            self.plot_win2.add_data_point("V", mcs * days_to_mcs, np.log10(self.sbml.ambersmithsimple['V']))


class CellularModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # set initial model parameters
        self.initial_uninfected = len(self.cell_list_by_type(self.U))
        self.ExtracellularVirus = self.sbml.ambersmithsimple['V']
        self.get_xml_element('virus_decay').cdata = self.sbml.ambersmithsimple['c'] * days_to_mcs

        if plot_CellModel:
            self.plot_win3 = self.add_new_plot_window(title='CPM Cells',
                                                      x_axis_title='days',
                                                      y_axis_title='Variables', x_scale_type='linear',
                                                      y_scale_type='linear',
                                                      grid=False, config_options={'legend': True})
            self.plot_win3.add_plot("U", style='Lines', color='blue', size=5)
            self.plot_win3.add_plot("I1", style='Lines', color='yellow', size=5)
            self.plot_win3.add_plot("I2", style='Lines', color='red', size=5)
            self.plot_win3.add_plot("D", style='Lines', color='purple', size=5)

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
            self.plot_win4.add_plot("V", style='Lines', color='blue', size=5)

            if overlay_AmbersModel:
                self.plot_win4.add_plot("AV", style='Dots', color='blue', size=5)

    def step(self, mcs):
        # Transition rule from U to I1
        secretor = self.get_field_secretor("Virus")
        for cell in self.cell_list_by_type(self.U):
            # Determine V from scalar virus from the ODE
            if how_to_determine_V == -1:
                b = self.sbml.ambersmithsimple['beta'] * self.sbml.ambersmithsimple['T0'] * days_to_mcs
                V = self.sbml.ambersmithsimple['V'] / self.sbml.ambersmithsimple['T0']

            # Determine V from scalar virus from the cellular model
            if how_to_determine_V == 0:
                b = self.sbml.ambersmithsimple['beta'] * self.initial_uninfected * days_to_mcs
                V = self.ExtracellularVirus / self.initial_uninfected

            # Determine V from the virus field
            if how_to_determine_V == 1:
                b = self.sbml.ambersmithsimple['beta'] * self.initial_uninfected * days_to_mcs
                uptake_probability = 0.0000001
                uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                V = abs(uptake.tot_amount) / uptake_probability
                secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

            # Calculate the probability of infection of individual cells based on the amount of virus PER cell
            p_UtoI1 = b * V
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

        # Production of extracellular virus
        secretor = self.get_field_secretor("Virus")
        V = self.ExtracellularVirus
        p = self.sbml.ambersmithsimple['p'] / self.initial_uninfected * self.sbml.ambersmithsimple['T0'] * days_to_mcs
        c = self.sbml.ambersmithsimple['c'] * days_to_mcs
        for cell in self.cell_list_by_type(self.I2):
            release = secretor.secreteInsideCellTotalCount(cell, p / cell.volume)
            self.ExtracellularVirus += release.tot_amount
        self.ExtracellularVirus -= c * V

        # Measure amount of extracellular virus field
        self.ExtracellularVirus_Field = 0
        for cell in self.cell_list:
            uptake_probability = 0.0000001
            uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
            V = abs(uptake.tot_amount) / uptake_probability
            self.ExtracellularVirus_Field += V
            secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

        if plot_CellModel:
            self.plot_win3.add_data_point("U", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.U)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I1", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I1)) / self.initial_uninfected)
            self.plot_win3.add_data_point("I2", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.I2)) / self.initial_uninfected)
            self.plot_win3.add_data_point("D", mcs * days_to_mcs,
                                          len(self.cell_list_by_type(self.DEAD)) / self.initial_uninfected)
            if how_to_determine_V == 1:
                self.plot_win4.add_data_point("V", mcs * days_to_mcs, np.log10(self.ExtracellularVirus_Field))
            else:
                self.plot_win4.add_data_point("V", mcs * days_to_mcs, np.log10(self.ExtracellularVirus))

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
