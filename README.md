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
        dof_string = "vscale*v + fscale*f_select/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "f_select = f + modify*boost;" #if energy is below threshold 'E', then add the boost potential 'fprime'
        dof_string+= "modify = step(E-energy);" #'modify' will be 1 when energy is below E
        dof_string+= "boost= 0.5 * k * (E - energy)^2;"
        
        
        self.addComputePerDof("v", dof_string); 
        #self.addComputePerDof("v", "v+dt*fprime/m; fprime=f*((1-modify) + modify*(alpha/(alpha+E-energy))^2); modify=step(E-energy)")
            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
        self.addComputePerDof("veloc", "v")
        
```
