import pandapower as pp
import pandapower.networks as pn
import numpy as np

# Step 1: Load the IEEE 33-bus system
def create_33_bus_system():
    net = pn.case33bw()
    return net

# Step 2: Calculate Loss Sensitivity Factors
def calculate_lsf(net):
    # Run a power flow analysis
    pp.runpp(net)
    
    # Extract line power losses (active power losses in MW)
    line_losses = net.res_line.pl_mw  # Active power loss on each line
    from_buses = net.line.from_bus.values
    print(from_buses)
    to_buses = net.line.to_bus.values
    
    # Calculate total power losses at each bus
    bus_losses = np.zeros(len(net.bus))
    for i, (from_bus, to_bus) in enumerate(zip(from_buses, to_buses)):
        bus_losses[from_bus] += line_losses[i]
        bus_losses[to_bus] += line_losses[i]
    
    # Active power injections at buses
    p_injected = net.res_bus.p_mw.values  # Active power injection at buses
    
    # Calculate LSF for each bus
    lsf = np.zeros(len(bus_losses))
    for i in range(len(bus_losses)):
        if p_injected[i] != 0:  # Avoid division by zero
            lsf[i] = bus_losses[i] / p_injected[i]
        else:
            lsf[i] = 0

    return lsf

# Step 3: Identify Optimal DG Placement
def optimal_dg_placement(lsf):
    # Sort buses by LSF (higher LSF indicates higher potential loss reduction)
    sorted_indices = np.argsort(lsf)[::-1]  # Descending order
    optimal_buses = sorted_indices[:5]  # Select top 5 buses for DG placement
    return optimal_buses

# Main script
if __name__ == "__main__":
    # Create the 33-bus system
    net = create_33_bus_system()
    
    # Calculate Loss Sensitivity Factors
    lsf = calculate_lsf(net)
    print("Loss Sensitivity Factors (LSF):", lsf)
    
    # Find optimal DG placement
    optimal_buses = optimal_dg_placement(lsf)
    print("Optimal buses for DG placement:", optimal_buses)

    
    print("Line Losses (MW):", net.res_line.pl_mw)
    print("Bus Losses (MW):", bus_losses)
    print("Power Injections (MW):", p_injected)
    print("Loss Sensitivity Factors (LSF):", lsf)
