import pandas as pd
import pandapower as pp
import pandapower.networks as pn

# Load the IEEE 33-bus test system
net = pn.case33bw()

# Define hourly load factors for a day (normalized)
hourly_load_factors = [
    0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15,
    1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65
]

# Base load values for each bus
base_loads = net.load.p_mw.copy()

# DataFrame to store LSF results for all buses and hours
lsf_results = pd.DataFrame(index=net.bus.index, columns=range(24))

# Loop through each hour
for t, load_factor in enumerate(hourly_load_factors):
    # Scale the loads according to the load factor
    net.load.p_mw = base_loads * load_factor
    
    # Run power flow for the hour
    pp.runpp(net)
    initial_losses = net.res_line.pl_mw.sum()
    
    # Calculate LSF for each bus (except slack)
    for bus in net.bus.index:
        if bus == net.ext_grid.bus.values[0]:  # Skip slack bus
            continue
        
        # Inject 1 MW at the bus
        pp.create_sgen(net, bus=bus, p_mw=1.0, q_mvar=0.0)
        pp.runpp(net)
        new_losses = net.res_line.pl_mw.sum()
        
        # Compute LSF
        lsf = (initial_losses - new_losses) / 1.0
        lsf_results.at[bus, t] = lsf
        
        # Remove the injected generator
        net.sgen.drop(net.sgen.index[-1], inplace=True)

# Convert all LSF values to numeric
lsf_results = lsf_results.apply(pd.to_numeric)
print(f"this is the lsf_results total: {lsf_results.sum(axis=1)}")

# Average LSF for each bus over 24 hours
average_lsf = lsf_results.mean(axis=1)

# Display top buses with the highest average LSF
top_buses = average_lsf.sort_values(ascending=False).head(10)
print("Top 10 Buses with the Highest Average LSF Over 24 Hours:")
print(top_buses)
