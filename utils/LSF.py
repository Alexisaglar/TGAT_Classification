import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Step 1: Load the IEEE 33-bus distribution system
net = pn.case33bw()

# List of load buses
load_buses = net.load.bus.values

# Number of buses and time intervals
num_buses = net.bus.shape[0]
num_hours = 24

# Step 2: Create a 24-hour load profile for each load bus
# For simplicity, we'll use a daily load profile that varies sinusoidally
time = np.arange(num_hours)
base_loads = net.load.p_mw.values.copy()

# Create a load profile: Base load * (0.5 + 0.5 * sin(pi * t / 12))
load_profile = 0.5 + 0.5 * np.sin(np.pi * time / 12)

# Initialize arrays to store LSFs
lsf_per_hour = np.zeros((num_hours, num_buses))

# Define a small perturbation value for active power (in MW)
delta_p = 0.1

# Step 3: Loop over each hour
for t in range(num_hours):
    print(f"Calculating for hour {t+1}/{num_hours}...")
    # Update the load values according to the profile
    net.load.p_mw = base_loads * load_profile[t]
    
    # Run power flow calculation
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
    
    # Calculate total system losses
    total_loss = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
    
    # Store the original load values
    original_loads = net.load.p_mw.copy()
    
    # Initialize array to store LSFs for this hour
    lsf_hour = np.zeros(num_buses)
    
    # Step 4: Calculate LSF for each bus
    for bus_idx in range(num_buses):
        # Check if the bus is a load bus
        if bus_idx in load_buses:
            # Perturb the active power at this bus
            net.load.loc[net.load.bus == bus_idx, 'p_mw'] += delta_p
            
            # Run power flow calculation after perturbation
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            
            # Calculate new total system losses
            new_total_loss = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
            print(net.res_trafo.pl_mw.sum())
            
            # Calculate the LSF: ΔP_loss / ΔP_i
            delta_loss = new_total_loss - total_loss
            lsf = delta_loss / delta_p
            lsf_hour[bus_idx] = lsf
            
            # Reset the load to original value
            net.load.loc[net.load.bus == bus_idx, 'p_mw'] -= delta_p
        else:
            # For non-load buses, LSF is zero
            lsf_hour[bus_idx] = 0.0
    
    # Store LSFs for this hour
    lsf_per_hour[t, :] = lsf_hour

# Step 5: Compute the time-averaged LSF for each bus
average_lsf = lsf_per_hour.mean(axis=0)

# Prepare the results
bus_indices = np.arange(num_buses)
result_df = pd.DataFrame({
    'Bus': bus_indices,
    'Average LSF': average_lsf
})

# Step 6: Output the results
print("\nTime-Averaged Loss Sensitivity Factors (LSF) for Each Bus:")
print(result_df)

# Optional: Plot the average LSFs
plt.figure(figsize=(12, 6))
plt.bar(bus_indices, average_lsf)
plt.xlabel('Bus Index')
plt.ylabel('Average LSF (MW Loss per MW Injection)')
plt.title('Time-Averaged Loss Sensitivity Factors for IEEE 33-Bus System')
plt.grid(True)
plt.show()
