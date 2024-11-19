import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt

def calculate_total_losses(net):
    """
    Calculate the total active power loss in the system.
    """
    pp.runpp(net)
    total_losses = net.res_line['pl_mw'].sum()
    return total_losses

def add_pv_system(net, bus_id, pv_power):
    """
    Add a PV system (static generator) to the specified bus in the network.
    """
    pp.create_sgen(net, bus=bus_id, p_mw=pv_power, q_mvar=0, name=f"PV_{bus_id}")

def reset_pv_systems(net):
    """
    Remove all existing PV systems (sgen elements) from the network.
    """
    net.sgen.drop(net.sgen.index, inplace=True)

def compare_pv_placement_per_bus_load(net):
    """
    Compare the impact of placing a PV system at each bus on total power losses.
    The PV power is equal to the total load of the bus.
    """
    # Calculate base-case losses
    base_loss = calculate_total_losses(net)
    print(f"Base-case total losses (no PV): {base_loss:.6f} MW\n")

    # Compare losses for PV placement at each bus
    results = {}
    for bus_id in net.bus.index:
        # Get the total load at the bus
        bus_load = net.load.loc[net.load['bus'] == bus_id, 'p_mw'].sum()
        if bus_load > 0:  # Only consider buses with load
            # Reset PV systems and add PV at the current bus
            reset_pv_systems(net)
            add_pv_system(net, bus_id, pv_power=bus_load)
            
            # Calculate total losses
            total_loss = calculate_total_losses(net)
            results[bus_id] = total_loss
            print(f"Total losses with PV (Load = {bus_load:.6f} MW) at Bus {bus_id}: {total_loss:.6f} MW")
        else:
            results[bus_id] = None  # No PV if no load at the bus

    # Reset to base case
    reset_pv_systems(net)
    return base_loss, results

def plot_results(base_loss, results):
    """
    Plot the impact of PV placement on total power losses.
    """
    buses = [bus for bus, loss in results.items() if loss is not None]
    losses = [loss for loss in results.values() if loss is not None]

    plt.figure(figsize=(10, 6))
    plt.bar(buses, losses, color='skyblue', label='With PV')
    plt.axhline(base_loss, color='red', linestyle='--', label='Base-case Loss')
    plt.xlabel('Bus Number')
    plt.ylabel('Total Power Losses (MW)')
    plt.title(f'Impact of PV Allocation on Total Losses (PV Power = Bus Load)')
    plt.legend()
    plt.grid()
    plt.show()

# Load the IEEE 33-bus system
net = pn.case33bw()

# Compare PV placement at all buses
base_loss, results = compare_pv_placement_per_bus_load(net)

# Find the best bus for PV placement
best_bus = min((bus for bus in results if results[bus] is not None), key=results.get)
print(f"\nThe best bus to place the PV system is Bus {best_bus}.\n")

# Plot the results
plot_results(base_loss, results)
