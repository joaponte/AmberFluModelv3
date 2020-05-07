
from cc3d.core.PySteppables import *
import numpy as np
# The purpose of this simulation is to show how we can take a pure ODE model and use it to build a numerically
# equivalent simulation in a multicellular framework
# We start by implementing the pure ODE model in Antimony (redefining the structure of parameters so that any parameters which depend
# on time-scale or total number of cells are broken out and are easy to compensate)
# We then define 1 MCS in wall clock time (here 1 minute, probably too short) and run the ODEs and check they
# reproduce the original paper results
# Next we go term by term in the original model and translate it into a homogeneous stochastic spatial model
# 
# First:In this case the equations include the rate of infection of suscptible cells (T) transitioning to early infected cells (I1)
# T->I1 as a function of T and the viral load V
# We use the viral load from the ODE and check that the number of T and I1 vs time agrees between the ODEs and spatial model
# We will use the time at which the number of Ts decreases by 5% as a metric for the agreement between the two models
# We don't expect exact agreement, because the stochastic spatial model predicts no infected cells for very low viral loads
# 
# Second: We next check that the transition from early infected to virus producing cells agrees with the ODE model.
# In this case we have I1->I2 as a function of I1
#
# Third, we include the death of I2 cells, I2->Dead at a rate that depends on I2 and a saturation term
# We quantify the agreement of the second and third sub models bu comparing the maximum value of I2 and the time
# at which I2 reaches its maximum relative to the time T starts to decrease.
# 
# Fourth, we include virus production based on the total number of I2 cells and check that the decay and production 
# rates agree--we expect divergence early when the number of I2 cells is small and discretization and stochasticity
# Have strong effects
# Our metric for agreement here is the MSD between the ODE and cellular viral loads
# ********************************************Not implemented Yet
# 
# Next, we  (not implemented yet) use the spatial model viral load rather than the ODE viral load to drive infection
# Next, we (not implemented yet) have the individual I2 cells produce virus rather than have the total count do it (no effect at this point, but significant later)
# Finally we replace the scalar viral load with a diffusing field viral load and check that we continue to satisfy 
# Previous constraints
#
# The first model is the base model from Amber's Paper
#
# Key difference is that she starts with some I1 cells
# Her I1 fraction is 7.5*10^-6 which is fewer than one cell for us....
# Probably the correct thing to do would be to run her simulation until the number of 
# infected cells reaches a threshold of one for us and then copy over the values of V
#
ModelString = '''        
        model ambersmithsimple()
        //Amber Smith Simple 4 parameter model
        T1model: ->T ; -beta * V * T
        I1model: ->I1 ; beta * V * T - k * I1
        Vmodel: -> V ; p * I2 - c * V
        I2model: -> I2 ; k * I1 - deltad * I2 / (Kdelta+I2)
        // I'm adding a count of the number of Dead Cells to make the bookkeeping easier
        Deadmodel: ->Dead ; deltad * I2 / (Kdelta+I2)
        //Parameters
        beta = 2.4 * 10^(-4);
        p = 1.6;
        c = 13.0;
        //c = 0.0 ; // set to zero temporarily to calibrate viral release use value above once production is calibrated
        k = 4.0;
        deltad0 = 1.6 * 10^6;
        T0 = 1.0*10^7; //I am putting the initial number of cells in explicity as a parameter
        //We need to rescale Kdelta to be a fraction of the initial number of susceptible cells and express all of our numbers in terms of this initial number
        Kdelta0 = 4.5*10^5
        Kratio= Kdelta0/T0
        Kdelta = Kratio*T0; // Original value of Kdelta was 4.5*10^5. I've rewritten it as a ratio to T0
        //Similarly, we need to rewrite deltad=deltad_ratio * T0 so that the death rate is independent of T0
        deltadratio=deltad0/T0
        deltad=deltadratio*T0
        
        //Initial Conditions
        
        T= 1.0*10^7;
        I1= 0.0; // orginal value was 75.0 Amber suggested swapping to 0
        I2 = 0.0;
        V = 75.0; // orginal value was 0.0 Amber suggested swapping to 75
        Dead = 0.0;
        end
'''
# The second model is a copy we can use to swap components in an out based on the model from Amber's Paper
ModelString2 = '''        
        model ambersmithsimple2()
        //Amber Smith Simple 4 parameter model
        T1model: ->T ; -beta * V * T
        I1model: ->I1 ; beta * V * T - k * I1
        Vmodel: -> V ; p * I2 - c * V
        I2model: -> I2 ; k * I1 - deltad * I2 / (Kdelta+I2)
        // I'm adding a count of the number of Dead Cells to make the bookkeeping easier
        Deadmodel: ->Dead ; deltad * I2 / (Kdelta+I2)
        
        //Parameters
        beta = 2.4 * 10^(-4);
        p = 1.6;
        c = 13.0;
        k = 4.0;
        deltad = 1.6 * 10^6;
        T0 = 1.0*10^7; //I am putting the initial number of cells in explicity as a parameter
        //We need to rescale Kdelta to be a fraction of the initial number of susceptible cells and express all of our numbers in terms of this initial number
        Kratio= 4.5*10^-2
        Kdelta = Kratio*T0; // Original value of Kdelta was 4.5*10^5. I've rewritten it as a ratio to T0
        
        //Initial Conditions
        
        T= 1.0*10^7;
        I1= 75.0;
        I2 = 0.0;
        V = 0.0;
        Dead = 0.0;
        end
'''

class AmberFluModelSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        # set spatial model parameters
        
        self.NInitialUninfected=float(len(self.cell_list)) # Scale factor for fraction of cells infected
        self.timescaleconversion =1440.0 # Amber's model is in days, current spatial model is in minutes 
        print("Number of initial cells",self.NInitialUninfected)
        self.ExtracellularVirus=75.0 # Corresponds to V in Amber's Model
        
        # Set up metrics for analysis
        # 
        # Keep track of beginning of infection time when I2 gets above 1% and T gets below 99%
        # Keep track of time at which I1 reaches its maximum and its maximum value (probably corrected by beginning of infection time above
        # Evaluate MSD between calculated and ODE viral load (again, probably lagged by the beginning of infection time)
        # Could also do MSD between I1, I2 and Dead
        #
        self.begininfectiontime=0.0
        self.begininfectionthreshold = 0.01 # express as fraction of total nuber of cells
        self.infectionbegun=False
        
        self.earlyinfectedmaxtime=0.0
        self.earlyinfectedmax=False
        self.earlyinfectedmaxvalue=0.0
        
        # set up same metrics for the cellular model
        self.cellbegininfectiontime=0.0
        self.cellbegininfectionthreshold = 0.01 # express as fraction of total nuber of cells
        self.cellinfectionbegun=False
        
        self.cellearlyinfectedmaxtime=0.0
        self.cellearlyinfectedmax=False
        self.cellearlyinfectedmaxvalue=0.0
        # Set up graphics windows
        
        
        self.add_free_floating_antimony(model_string=ModelString, model_name='ambersmithsimple', step_size=1.0/self.timescaleconversion)
        self.add_free_floating_antimony(model_string=ModelString2, model_name='ambersmithsimple2', step_size=1.0/self.timescaleconversion)
        
        self.plot_win = self.add_new_plot_window(title='Amber Smith Model',
                                                 x_axis_title='Minutes',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)

        self.plot_win.add_plot("T", style='Lines', color='red', size=5)
        
        self.plot_win.add_plot("I1", style='Lines', color='orange', size=5)
        self.plot_win.add_plot("I2", style='Lines', color='green', size=5)
        self.plot_win.add_plot("Dead", style='Lines', color='blue', size=5)
        self.plot_win2 = self.add_new_plot_window(title='Amber Smith Model Virus Comparison',
                                                 x_axis_title='Minutes',
                                                 y_axis_title='Virus', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        self.plot_win2.add_plot("V", style='Lines', color='blue', size=5)
        self.plot_win2.add_plot("VCellular", style='Dots', color='green', size=5)
        self.plot_win3 = self.add_new_plot_window(title='Cellularized Amber Smith Model',
                                                 x_axis_title='Minutes',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        self.plot_win3.add_plot("Uninfected", style='Lines', color='red', size=5)
        
        self.plot_win3.add_plot("I1", style='Lines', color='orange', size=5)
        self.plot_win3.add_plot("I2", style='Lines', color='green', size=5)
        self.plot_win3.add_plot("Dead", style='Dots', color='blue', size=5)
        
        self.plot_win4 = self.add_new_plot_window(title='Lagged Difference Between Models',
                                                 x_axis_title='Lagged Minutes',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        self.plot_win4.add_plot("Uninfected", style='Lines', color='red', size=5)
        
        self.plot_win4.add_plot("I1", style='Lines', color='orange', size=5)
        self.plot_win4.add_plot("I2", style='Lines', color='green', size=5)
        self.plot_win4.add_plot("Dead", style='Dots', color='blue', size=5)
    
    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

        self.timestep_sbml()
        # I am rescaling all of these by the initial number of uninfected cells T0 in the Antimony
        self.plot_win.add_data_point("T", mcs, self.sbml.ambersmithsimple['T']/self.sbml.ambersmithsimple['T0'])
        self.plot_win.add_data_point("I1", mcs, self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'])
        self.plot_win.add_data_point("I2", mcs, self.sbml.ambersmithsimple['I2']/self.sbml.ambersmithsimple['T0'])
        self.plot_win.add_data_point("Dead", mcs, self.sbml.ambersmithsimple['Dead']/self.sbml.ambersmithsimple['T0'])
        self.plot_win2.add_data_point("V", mcs, self.sbml.ambersmithsimple['V'])
        
        
        # Check if the serious infection has started and update flags and values
        if not self.infectionbegun:
            if self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'] > self.begininfectionthreshold:
                self.infectionbegun=True
                self.begininfectiontime=mcs # converting Amber's time to MCS units
                print("Infection started at",self.begininfectiontime)
        
        # Check if I1 has reached maximum and update flags and values
        if self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'] > self.earlyinfectedmaxvalue:
            self.earlyinfectedmaxvalue=self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'] 
            self.earlyinfectedmaxtime=mcs # converting Amber's time to MCS units
        else:
            if mcs>100 and self.sbml.ambersmithsimple['I1']/self.sbml.ambersmithsimple['T0'] > 0.2:
                print("I2 max time", self.earlyinfectedmaxtime, "I2 max value", self.earlyinfectedmaxvalue )
                print("I2 max time after infection start", self.earlyinfectedmaxtime-self.begininfectiontime)
        # Iterate over all uninfected cells and convert them to infect1 with probability V*beta
        pconvert=self.sbml.ambersmithsimple['V']*self.sbml.ambersmithsimple['beta']/self.timescaleconversion/5.0
        # print(pconvert) # Check that pconvert makes sense
        for cell in self.cell_list_by_type(self.U):
            if np.random.random() < pconvert:                          
                cell.type=self.I1
        
        # Iterate over all infected1 cells and convert them to infected2 with probability k
        pconvert2=self.sbml.ambersmithsimple['k']/self.timescaleconversion# Had/5.0 before
        for cell in self.cell_list_by_type(self.I1):
            if np.random.random() < pconvert2:                          
                cell.type=self.I2
                
        # Repeat checks for cellular model Check if the serious infection has started and update flags and values
        if not self.cellinfectionbegun:
            if float(len(self.cell_list_by_type(self.I1)))/float(self.NInitialUninfected) > self.cellbegininfectionthreshold:
                self.cellinfectionbegun=True
                self.cellbegininfectiontime=mcs # Using CC3D time units
                print("Cellular Infection started at",self.cellbegininfectiontime)
        
        # Check if number of early infected cells has reached maximum and update flags and values
        if float(len(self.cell_list_by_type(self.I1)))/float(self.NInitialUninfected) > self.cellearlyinfectedmaxvalue:
            self.cellearlyinfectedmaxvalue=float(len(self.cell_list_by_type(self.I1)))/float(self.NInitialUninfected) 
            self.cellearlyinfectedmaxtime=mcs # Using CC3D time units
        else:
            if mcs>100 and float(len(self.cell_list_by_type(self.I1)))/float(self.NInitialUninfected) > 0.2:
                print("Cellular I2 max time", self.cellearlyinfectedmaxtime, "Cell I2 max value", self.cellearlyinfectedmaxvalue )
                print("Cellular I2 max time after infection start", self.cellearlyinfectedmaxtime-self.cellbegininfectiontime)
                
        # Iterate over all infected2 cells and convert them to dead with probability deltad/(Kdelta+I2)
        # Note that the value of Kdelta needs to be changed since I2 is scaled differently and the saturation
        # is not yet implemented
        Kdelta=self.NInitialUninfected*self.sbml.ambersmithsimple['Kratio'] # Gives rescaled value of Kdelta for Michaelis saturation
        pconvert3=(self.sbml.ambersmithsimple['deltadratio']*self.NInitialUninfected/(Kdelta+len(self.cell_list_by_type(self.I2))))/self.timescaleconversion
        #pconvert3=0
        for cell in self.cell_list_by_type(self.I2):
            if np.random.random() < pconvert3:                          
                cell.type=self.DEAD     
        
        # Model of virus production--initially will have all I2 cells produce equally, later have each I2 cell produce separately
        # Should be expressed as a tiny SBML model or as a secretor and a viral load field with decay
        self.ExtracellularVirus+=len(self.cell_list_by_type(self.I2))*self.sbml.ambersmithsimple['p']/2.0#/self.timescaleconversion
        # Model of virus decay
        # Should be expressed as a tiny SBML model or as a secretor and a viral load field with decay
        self.ExtracellularVirus-=self.ExtracellularVirus*self.sbml.ambersmithsimple['c']/self.timescaleconversion
        
        
        # Plot values
        
        self.plot_win2.add_data_point("VCellular", mcs, self.ExtracellularVirus)
        
        self.plot_win3.add_data_point("Uninfected", mcs, len(self.cell_list_by_type(self.U))/self.NInitialUninfected)
        
        self.plot_win3.add_data_point("I1", mcs, len(self.cell_list_by_type(self.I1))/self.NInitialUninfected)
        self.plot_win3.add_data_point("I2", mcs, len(self.cell_list_by_type(self.I2))/self.NInitialUninfected)
        self.plot_win3.add_data_point("Dead", mcs, len(self.cell_list_by_type(self.DEAD))/self.NInitialUninfected)
        
        
        # Plot lagged differences between cell populations
        # Start when both populations are infected
        # Josh--you will need to save the time series in a set of lists so you can do this
        # Once both times series have had infection begin
        # Plot (x(t-starttime)-X(t-cellstarttime)) for each series
        
        # Could do same thing to show virus with lags and also RMS deviation with lags
        
        #Next step have cells produce virus
        #Next step, have virus diffuse and cells infected by viral field rather than the external variable


        
        
        
        

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


        