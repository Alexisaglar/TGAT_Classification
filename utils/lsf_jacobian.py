import pandapower as pp
import pandapower.networks as pn
import numpy as np

# Load the case33bw network from pandapower
net = pn.case33bw()

# Function to calculate the loss sensitivity factor
def calculate_loss_sensitivity(net):
    # Run the power flow to get the required data
    pp.runpp(net)

    # Extract necessary data from the power flow results
    bus_data = net.res_bus
    line_data = net.line
    # Ybus, _, _ = pp.create_Ybus(net, net._options["pd_net"])

    # Number of buses
    num_buses = len(bus_data)
    
    # Initialize variables
    P = bus_data["p_mw"].values  # Active power injections (MW)
    Q = bus_data["q_mvar"].values  # Reactive power injections (MVAR)
    V = bus_data["vm_pu"].values  # Voltage magnitudes (p.u.)
    delta = bus_data["va_degree"].values * np.pi / 180  # Voltage angles (rad)

    # Extract Zbus from the Ybus matrix
    Zbus = np.linalg.inv(Ybus.toarray())
    rij = np.real(Zbus)
    xij = np.imag(Zbus)

    # Initialize loss sensitivity factors
    alpha = np.zeros(num_buses)

    # Calculate loss sensitivity factors
    for i in range(num_buses):
        for j in range(num_buses):
            a_ij = (rij[i, j] / (V[i] * V[j])) * np.cos(delta[i] - delta[j])
            b_ij = (rij[i, j] / (V[i] * V[j])) * np.sin(delta[i] - delta[j])
            alpha[i] += a_ij * P[j] - b_ij * Q[j]
        alpha[i] *= 2  # Multiply by 2 as per the formula

    return alpha

# Calculate loss sensitivity factors
loss_sensitivity_factors = calculate_loss_sensitivity(net)

# Display results
for i, alpha_i in enumerate(loss_sensitivity_factors):
    print(f"Loss sensitivity factor for bus {i + 1}: {alpha_i:.6f}")
