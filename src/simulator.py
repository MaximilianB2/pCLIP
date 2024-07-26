import matplotlib.pyplot as plt
from model import ChemicalReactor

import matplotlib.pyplot as plt

def plot_results(history):
    time = [state['time'] for state in history]
    Cb = [state['Cb'] for state in history]
    V = [state['V'] for state in history]
    Cb_setpoint = [state['Cb_setpoint'] for state in history]
    V_setpoint = [state['V_setpoint'] for state in history]
    Tc = [state['Tc'] for state in history]
    Fin = [state['Fin'] for state in history]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

    # Concentration A plot
    ax1.plot(time, Cb, label='Cb')
    ax1.plot(time, Cb_setpoint, 'r--', label='Cb Setpoint')
    ax1.set_ylabel('Concentration B (mol/m^3)')
    ax1.set_title('Concentration B vs Time')
    ax1.legend()

    # Volume plot
    ax2.plot(time, V, label='V')
    ax2.plot(time, V_setpoint, 'r--', label='V Setpoint')
    ax2.set_ylabel('Volume (m^3)')
    ax2.set_title('Volume vs Time')
    ax2.legend()

    # Cooling temperature plot
    ax3.step(time, Tc, where='post', label='Tc')
    ax3.set_ylabel('Cooling Temperature (K)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Cooling Temperature vs Time')
    ax3.legend()

    # Inlet flow rate plot
    ax4.step(time, Fin, where='post', label='Fin')
    ax4.set_ylabel('Inlet Flow Rate (m^3/s)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Inlet Flow Rate vs Time')
    ax4.legend()

    plt.tight_layout()
    plt.show()

initial_state = [1, 0, 0, 295, 100]  # [Ca, Cb, Cc, T, V]
initial_inputs = [300, 100]  # [Tc, Fin]
unstable_gains = [100, 100, 0, 1, 0, 0]  # Unstable gains
reactor = ChemicalReactor(initial_state, initial_inputs)
reactor.set_pid_gains(unstable_gains)
reactor.set_setpoints([0.8, 101])  # New setpoints for Ca and V

# Run simulation
unstable_history = reactor.run_simulation(duration=10)  # 1000 seconds
plot_results(unstable_history)