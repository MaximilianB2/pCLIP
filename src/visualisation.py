import matplotlib.pyplot as plt
from model import ChemicalReactor
import json
import re
import numpy as np
from matplotlib.colors import to_rgba

def plot_results(histories, labels):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

    # Find the maximum time across all histories
    max_time = max(max(state['time'] for state in history) for history in histories)

    # Define base colors for each variable
    colors = {
        'Cb': 'blue',
        'V': 'green',
        'Tc': 'red',
        'Fin': 'purple'
    }

    # Calculate alpha values for each history (darker for more recent)
    alphas = [0.05] * (len(histories) - 2) + [0.5, 0.9]

    for history, label, alpha in zip(histories, labels, alphas):
        time = [state['time'] for state in history]
        Cb = [state['Cb'] for state in history]
        V = [state['V'] for state in history]
        Tc = [state['Tc'] for state in history]
        Fin = [state['Fin'] for state in history]

        # Concentration B plot
        ax1.plot(time, Cb, color=to_rgba(colors['Cb'], alpha), label=f'Cb ({label})')
        
        # Volume plot
        ax2.plot(time, V, color=to_rgba(colors['V'], alpha), label=f'V ({label})')
        
        # Cooling temperature plot
        ax3.plot(time, Tc, color=to_rgba(colors['Tc'], alpha), label=f'Tc ({label})')
        
        # Inlet flow rate plot
        ax4.plot(time, Fin, color=to_rgba(colors['Fin'], alpha), label=f'Fin ({label})')
    ax1.axhline(y=0.8, color='black', linestyle='--', label='Cb Setpoint')
    ax2.axhline(y=101, color='black', linestyle='--', label='V Setpoint')
    # Set titles, labels, and limits
    ax1.set_title('Concentration B vs Time')
    ax1.set_ylabel('Concentration B (mol/m^3)')
    ax1.set_xlabel('Time (s)')
    ax1.set_xlim(0, max_time)
    # ax1.legend()

    ax2.set_title('Volume vs Time')
    ax2.set_ylabel('Volume (m^3)')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim(0, max_time)
    # ax2.legend()

    ax3.set_title('Cooling Temperature vs Time')
    ax3.set_ylabel('Cooling Temperature (K)')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, max_time)
    # ax3.legend()

    ax4.set_title('Inlet Flow Rate vs Time')
    ax4.set_ylabel('Inlet Flow Rate (m^3/s)')
    ax4.set_xlabel('Time (s)')
    ax4.set_xlim(0, max_time)
    # ax4.legend()

    plt.tight_layout()
    plt.savefig('convergence.pdf')
    plt.show()

# Load and parse the JSON data
with open('./interactions/llm_interactions_20240726_164019.json', 'r') as f:
    data = [json.loads(line) for line in f]


# Extract suggested gains
suggested_gains_list = []
for item in data:
    suggestion = item['suggestion']
    suggested_gain = re.search(r'Suggested gains: \[([\d., ]+)\]', suggestion)
    if suggested_gain:
        suggested_gains_list.append([float(x) for x in suggested_gain.group(1).split(', ')])

# Initial conditions
initial_state = [1, 0, 0, 295, 100]  # [Ca, Cb, Cc, T, V]
initial_inputs = [300, 100]  # [Tc, Fin]
setpoints = [0.8, 101]  # Setpoints for Cb and V
unstable_gains = [50, 0, 0, 6, 0, 10]
# Run simulations for each set of suggested gains
histories = []
labels = []
# First, simulate with unstable gains
reactor = ChemicalReactor(initial_state, initial_inputs)
reactor.set_setpoints(setpoints)
reactor.set_pid_gains(unstable_gains)
history = reactor.run_simulation(duration=10)  # 10 seconds
histories.append(history)
labels.append('Unstable Gains')

for i, gains in enumerate(suggested_gains_list):
    # Reset the reactor for each simulation
    reactor = ChemicalReactor(initial_state, initial_inputs)
    reactor.set_setpoints(setpoints)
    reactor.set_pid_gains(gains)
    
    history = reactor.run_simulation(duration=10)  # 10 seconds
    histories.append(history)
    labels.append(f'Gains {i+1}')

# Plot results
plot_results(histories, labels)


