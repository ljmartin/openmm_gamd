implementing GaMD in openmm with a LangevinIntegrator


GaMD in OpenMM

The most convenient way to implement GaMD in OpenMM is using a custom integrator. In the custom integrator, the force has already been calculated by OpenMM. That means we can't edit the potential energy function and expect to re-calculate the force. Instead, we find out what the change in potential energy would have done to the force, and do the same thing within the integrator.

Normally, the force is the negative of the gradient of the potential 

$V(\vec r)$: $$f = -1 \left( \frac{\partial V(\vec r)}{\partial \vec r}\right)$$

In GaMD, we use a modified potential:
$$V'(\vec r) = V(\vec r) + dV(\vec r)$$

where:
$$dV(\vec r) = \frac{k}{2} (E - V(\vec r))^2$$

So the modified force is now:
$$f' = -1 \left( \frac{\partial V(\vec r) + \frac{k}{2} (E - V(\vec r))^2}{\partial \vec r}\right)$$

Separate into two derivatives with the Sum rule:
$$f' = - \left( \frac{\partial V(\vec r)}{\partial \vec r}\right) - \left( \frac{\partial \frac{k}{2} (E - V(\vec r))^2 }{\partial \vec r}\right) $$

Now to start simplifying. The left hand term is already $-f$ and we can move $\frac{k}{2}$ outside (Constant rule):
$$f' = f - \frac{k}{2} \left( \frac{\partial (E - V(\vec r))^2 }{\partial \vec r}\right) $$

Expanding the right hand side:
$$f' = f - \frac{k}{2} \left( \frac{\partial (E^2 - 2EV(\vec r) + V(\vec r)^2) }{\partial \vec r}\right) $$

$E^2$ is a constant and doesn't affect the derivative, so remove it:
$$f' = f - \frac{k}{2} \left( \frac{\partial ( - 2EV(\vec r) + V(\vec r)^2) }{\partial \vec r}\right) $$

Factor out one of the $V(\vec r)$ terms so we can do a chain rule afterwards:
$$f' = f - \frac{k}{2} \left( \frac{\partial ( V(\vec r)(- 2E + V(\vec r)) }{\partial \vec r}\right) $$

Now use the chain rule:
$$f' = f - \frac{k}{2} \left( \frac{\partial V(\vec r) }{\partial \vec r}\cdot(-2E + V(\vec r)) + \frac{\partial (-2E + V(\vec r)) }{\partial \vec r} \cdot V(\vec r) \right) $$

This is where life gets easier. The left derivative is $-f$ again. The $-2E$ can be removed from the right derivative, since it's a constant, meaning the right derivative is just $-f$ as well!
$$f' = f - \frac{k}{2} \left( -f\cdot(-2E + V(\vec r)) -fV(\vec r) \right) $$

One step at a time...
$$f' = f - \frac{k}{2} \left( 2fE - fV(\vec r) -fV(\vec r) \right) $$

then $$f' = f - \frac{k}{2} \left( 2fE - 2fV(\vec r) \right) $$

then $$f' = f - \frac{k}{2} \left( 2f(E - V(\vec r)) \right) $$

then $$f' = f - fk \left( E - V(\vec r) \right) $$

So finally, we have the modified force being the original force multiplied by a factor of $k$, $E$, and $V(\vec r)$:
$$f' = f \cdot (1 - k(E - V(\vec r) )$$

The custom integrator below is a Langevin integrator with some extra parameters like $E$ and $k$ (one of each for the dihedrals force group, and one of each for everything else, which Miao et al. call 'total' potential), which are used to calculate the modified force.

It also has methods to calculate the magnitude of the boost, which can be used later to re-weight the free energy estimates.



# the integrator:

```python
class CustomGaMDLangevinIntegrator(CustomIntegrator):
    def __init__(self, temperature, friction, dt, ktot, Etot, kgrp, Egrp, forceGroup):
        self.ktot = ktot 
        self.Etot = Etot 
        self.kgrp = kgrp
        self.Egrp = Egrp
        self.forceGroup = str(forceGroup)
        
        CustomIntegrator.__init__(self, dt)
            #lew added:
        self.addGlobalVariable("ktot", self.ktot)
        self.addGlobalVariable("Etot", self.Etot)
        self.addGlobalVariable("kgrp", self.ktot)
        self.addGlobalVariable("Egrp", self.Egrp)
        self.addGlobalVariable("groupEnergy", 0)
        
            #normal langevin:  
        self.addGlobalVariable("temperature", temperature);
        self.addGlobalVariable("friction", friction);
        self.addGlobalVariable("vscale", 0);
        self.addGlobalVariable("fscale", 0);
        self.addGlobalVariable("noisescale", 0);
        self.addPerDofVariable("x0", 0);
        
        self.addPerDofVariable("fgrp", 0)
        
            #normal langevin:                                                                  
        self.addUpdateContextState();
        
        self.addComputeGlobal("groupEnergy", "energy"+self.forceGroup)
        self.addComputePerDof("fgrp", "f"+self.forceGroup)
        
        self.addComputeGlobal("vscale", "exp(-dt*friction)");
        self.addComputeGlobal("fscale", "(1-vscale)/friction");
        #original line:                
        self.addComputeGlobal("noisescale", "sqrt(kT*(1-vscale*vscale)); kT=0.00831451*temperature");
        self.addComputePerDof("x0", "x");
            #original langevin line:                                                                                      
        #self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)");  
            #GaMD:
        dof_string = "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "fprime= fprime1 + fprime2;"
        #fprime2 is the dihedral force modified by the boost. Boot calculated using group only. 
        dof_string+= "fprime2 = fgrp*((1-modifyGroup) + modifyGroup* (1 - kgrp*(Egrp - groupEnergy)) ) ;"
        #fprime1 is the other forces modified by the boost, but the boost is calculated using TOTAL energy. 
        dof_string+= "fprime1 = ftot*((1-modifyTotal) + modifyTotal* (1 - ktot*(Etot - energy)) );"
        
        dof_string+= "ftot=f-fgrp;"
        dof_string+= "modifyGroup=step(Egrp-groupEnergy);"
        dof_string+= "modifyTotal=step(Etot-energy);"
        self.addComputePerDof("v", dof_string); 
            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
        self.addComputePerDof("veloc", "v")
        
    def setKtot(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('ktot', newK)
        
    def setEtot(self, newE):
        if not is_quantity(newE):
            newE = newE*kilojoules_per_mole
        self.setGlobalVariableByName('Etot', newE)
        
    def setKgrp(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('kgrp', newK)
        
    def setEgrp(self, newE):
        #if not is_quantity(newE):
        #    newE = newE*kilojoules_per_mole
        #    print(newE)
        self.setGlobalVariableByName('Egrp', newE)
          
    def getGrpBoost(self, grpEnergy):
        kgrp = self.getGlobalVariableByName('kgrp')/kilojoules_per_mole
        Egrp = self.getGlobalVariableByName('Egrp')*kilojoules_per_mole
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        if (grpEnergy > Egrp):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * kgrp * (Egrp-grpEnergy)**2 ) # 'k' parameter should instead be per kj/mol
    
    def getTotBoost(self, totEnergy):
        ktot = self.getGlobalVariableByName('ktot')/kilojoules_per_mole
        Etot = self.getGlobalVariableByName('Etot')*kilojoules_per_mole
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        if (totEnergy > Etot):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * ktot * (Etot-totEnergy)**2 ) # 'k' parameter should instead be per kj/mol
        
    def getEffectiveEnergy(self, totEnergy, grpEnergy):
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        
        group_boost = self.getGrpBoost(grpEnergy)
        total_boost = self.getTotBoost(totEnergy)
        
        return totEnergy + group_boost + total_boost
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
