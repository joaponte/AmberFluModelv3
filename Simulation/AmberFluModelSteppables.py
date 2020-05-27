from cc3d.core.PySteppables import *
import numpy as np
import os
import Parameters

Data_writeout_ODEs = False
Data_writeout_CellularModel = True

## How to determine V
# -1 pulls from the scalar virus from the ODE original model (no feedback in the cellular model)
#  0 pulls from the scalar virus from the cellular model (feedback in the cellular model but no field)
#  1 pulls from the virus field
how_to_determine_V = 1

min_to_mcs = 10.0  # min/mcs
days_to_mcs = min_to_mcs / 1440.0  # day/mcs
days_to_simulate = 10.0 #10 in the original model

production_multiplier = Parameters.P
diffusion_multiplier = Parameters.D
replicate = Parameters.R

'''Smith AP, Moquin DJ, Bernhauerova V, Smith AM. Influenza virus infection model with density dependence 
supports biphasic viral decay. Frontiers in microbiology. 2018 Jul 10;9:1554.'''

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

        # Adding free floating coinfection antimony model
        self.add_free_floating_antimony(model_string=Coinfection_ModelString, model_name='coinfection',
                                        step_size=days_to_mcs)
        # Changing initial values according to discussions with Amber Smith
        state = {}
        state['I1'] = 0.0
        state['VA'] = 75.0
        state['VB'] = 75.0
        self.set_sbml_state(model_name='coinfection', state=state)

    def step(self, mcs):
        self.timestep_sbml()

class CellularModelSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # set initial model parameters
        self.initial_uninfected = len(self.cell_list_by_type(self.U))
        self.ExtracellularVirus = self.sbml.coinfection['VA']
        self.get_xml_element('virus_decay').cdata = self.sbml.coinfection['c'] * days_to_mcs
        self.ExtracellularVirusB = self.sbml.coinfection['VB']
        self.get_xml_element('virusB_decay').cdata = self.sbml.coinfection['c'] * days_to_mcs

    def step(self, mcs):
        # Transition rule from U to I1
        secretor = self.get_field_secretor("Virus")
        secretorB = self.get_field_secretor("VirusB")
        for cell in self.cell_list_by_type(self.U):
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
        p *= production_multiplier
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

class Data_OutputSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        if Data_writeout_CellularModel:
            folder_path = '/Users/Josua/Downloads/AmberFluModelv3/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_name2 = 'cellularizedmodel_%.5d_%.5d_%i.txt' % (
                production_multiplier*100,diffusion_multiplier*100,replicate)
            self.output2 = open(folder_path + file_name2, 'w')
            self.output2.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                'Time', 'U', 'AI1', 'AI2', 'AD', 'BI2', 'BI2', 'BD', 'VA', 'VB'))
            self.output2.flush()

    def step(self, mcs):
        if Data_writeout_CellularModel:
            # Record variables from Cellularized Model
            d = mcs * days_to_mcs
            U = len(self.cell_list_by_type(self.U))
            I1 = len(self.cell_list_by_type(self.I1))
            I2 = len(self.cell_list_by_type(self.I2))
            DA = len(self.cell_list_by_type(self.DEAD))
            I1B = len(self.cell_list_by_type(self.I1B))
            I2B = len(self.cell_list_by_type(self.I2B))
            DB = len(self.cell_list_by_type(self.DEADB))

            self.Virus_Field = 0
            secretor = self.get_field_secretor("Virus")
            for cell in self.cell_list:
                uptake_probability = 0.0000001
                uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                V = abs(uptake.tot_amount) / uptake_probability
                self.Virus_Field += V
                secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

            self.Virus_FieldB = 0
            secretor = self.get_field_secretor("VirusB")
            for cell in self.cell_list:
                uptake_probability = 0.0000001
                uptake = secretor.uptakeInsideCellTotalCount(cell, 1E6, uptake_probability)
                V = abs(uptake.tot_amount) / uptake_probability
                self.Virus_FieldB += V
                secretor.secreteInsideCellTotalCount(cell, abs(uptake.tot_amount) / cell.volume)

            self.output2.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
            d, U, I1, I2, DA, I1B, I2B, DB, self.Virus_Field, self.Virus_FieldB))
            self.output2.flush()

    def finish(self):
        if Data_writeout:
            self.output.close()