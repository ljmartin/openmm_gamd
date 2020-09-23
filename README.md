implementing GaMD in openmm with a LangevinIntegrator


# the integrator:

```python
class CustomGaMDLangevinIntegrator(CustomIntegrator):
    def __init__(self, temperature, friction, dt, k, E):
        self.k = k 
        self.E = E #threshold value
        
        CustomIntegrator.__init__(self, dt)
            #lew added:
        self.addGlobalVariable("k", k)
        self.addGlobalVariable("E", E)
            #normal langevin:  
        self.addGlobalVariable("temperature", temperature);
        self.addGlobalVariable("friction", friction);
        self.addGlobalVariable("vscale", 0);
        self.addGlobalVariable("fscale", 0);
        self.addGlobalVariable("noisescale", 0);
        self.addPerDofVariable("x0", 0);
            #normal langevin:                                                                  
        self.addUpdateContextState();
        self.addComputeGlobal("vscale", "exp(-dt*friction)");
        self.addComputeGlobal("fscale", "(1-vscale)/friction");
        #original line:                
        self.addComputeGlobal("noisescale", "sqrt(kT*(1-vscale*vscale)); kT=0.00831451*temperature");
        self.addComputePerDof("x0", "x");
            #original langevin line:                                                                                      
        #self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)");  
            #GaMD:
        #self.addComputePerDof("v", "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);fprime=f*((1-modify) + modify*(alpha/(alpha+E-energy))^2);modify=step(E-energy)"); 
        dof_string = "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "fprime = f*((1-modify) + modify*((energy + 0.5*k*(E-energy )^2)/energy)  );" #multiplying f by factor equivalent to what V(r) was multiplied by
        dof_string+= "modify = step(E-energy);" #'modify' will be 1 when energy is below E

        
        self.addComputePerDof("v", dof_string); 
        #self.addComputePerDof("v", "v+dt*fprime/m; fprime=f*((1-modify) + modify*(alpha/(alpha+E-energy))^2); modify=step(E-energy)")
            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
        self.addComputePerDof("veloc", "v")
        
    def getEffectiveEnergy(self, energy):
        """Given the actual potential energy of the system, return the value of the effective potential."""
        k = self.k
        E = self.E
        if not is_quantity(energy):
            energy = energy*kilojoules_per_mole # Assume kJ/mole
        if (energy > E):
            return energy
        return energy + ( 0.5 * k * (E-energy)**2 ) # 'k' parameter should instead be per kj/mol

```


now, when setting `E` to `Vmax`, estimate GaMD parameters like this:

```python
pe = np.array(my_potential_energies)

#set the desired maximum standard deviation of the boost potential to be 10kT: 
sigma_0 = (MOLAR_GAS_CONSTANT_R * TEMPERATURE ).value_in_unit(kilojoule_per_mole) * 10
print(f'Sigma0: {sigma_0}')

#potential energy statistics:
Vmax = pe.max()
Vmin = pe.min()
Vavg = pe.mean()
Vstd = np.std(pe)

print(f'Vmax: {Vmax},\nVmin: {Vmin},\nVavg: {Vavg},\nVstd: {Vstd}')
k_0 = min(1, sigma_0/Vstd * ((Vmax-Vmin)/(Vmax-Vavg)))

k = k_0 * (1 / (Vmax - Vmin) )

print(f'k_0: {k_0},\nk: {k}')
```
