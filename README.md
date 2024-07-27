# pCLIP: Process Control Language Interface Program 📎

## 🌟 Overview

This repository contains a simulation of a chemical reactor system with PID (Proportional-Integral-Derivative) control, enhanced by Large Language Model (LLM) assistance for tuning. The project represents a first step towards easier communication with chemical plants, aiming to bridge the gap between human operators and complex industrial processes. It demonstrates advanced control strategies in chemical engineering and showcases the potential of AI-assisted optimization in industrial settings.

## 🔧 Key Components

1. **ChemicalReactor Class**: Implements a dynamic model of a chemical reactor using CasADi for efficient numerical integration.

2. **LLMOperatorInterface Class**: Provides an interface for human operators to interact with the simulation and receive AI-assisted suggestions for PID gain tuning, simulating enhanced plant-operator communication.

3. **PID Control**: Implements PID control for maintaining desired concentration (Cb) and volume (V) setpoints.

4. **LLM Integration**: Utilizes Claude, an AI language model, to analyze system performance and suggest PID gain adjustments, mimicking an intelligent assistant for plant operators.

## ✨ Features

- 🔄 Real-time simulation of chemical reactions and reactor dynamics
- 🎛️ Interactive PID tuning with immediate visual feedback
- 🤖 AI-assisted gain optimization using state-of-the-art language models
- 📊 Comprehensive performance metrics for control quality assessment
- 📈 Data logging and visualization tools for analysis
- 💬 Natural language interface for easier interaction with the simulated plant
- 🧠 Explanatory feedback on control actions, enhancing operator understanding and confidence

## 🔬 Technical Details

- **Simulation Model**: Incorporates reaction kinetics, heat transfer, and mass balance equations
- **Numerical Methods**: Employs 4th order Runge-Kutta (RK4) integration for accurate ODE solving
- **Control Implementation**: Utilizes position-form PID control with anti-windup mechanisms
- **LLM Integration**: Leverages the Claude API for generating context-aware tuning suggestions and explanations

## 🚀 Installation and Usage

1. Clone the repository:

2. Install the required dependencies:
_
With pip:
``pip install --requirments.txt``

With conda:
``conda env create --name llm_operator --file environment.yml``

3. Set up your Claude API key:
- Obtain an API key from Anthropic
- Set the environment variable:
  ```
  export CLAUDE_API_KEY='your_api_key_here'
  ```

## 🖥️ How to Use

1.. **Starting the Simulation**: 
- Run `llm_operator.py` to start the Chemical Reactor PID Tuning Interface.
- The initial unstable gains will be displayed.

2. **Interacting with the Simulation**:
- Enter new gain values (comma-separated) when prompted or ...
- Type 'a' for AI suggestion.
- Type 'q' to quit the simulation.

3. **Using AI Suggestions**:
- When you type 'a', the LLM will analyze the current performance and suggest new gains.
- The suggestion includes explanations for both Cb and V controllers.
- You can choose to apply these suggested gains or not.

4. **Visualizing Results**:
- After each simulation run, plots will be displayed showing the system's performance.
- These plots include Concentration of B, Volume, Cooling Temperature, and Inlet Flow Rate over time.

5. **Interpreting LLM Feedback**:
- The LLM provides explanations for its suggestions, helping you understand the reasoning behind the proposed changes.
- Use this information to make informed decisions about adjusting the PID gains.

6. **Iterative Tuning**:
- Continue adjusting gains or requesting AI suggestions until you achieve satisfactory performance.
- Pay attention to the performance metrics provided after each simulation run.

Remember, the goal is to stabilize both the Concentration B (Cb) and Volume (V) at their respective setpoints while minimizing oscillations and steady-state errors.

## 🔮 Future Work

- Integration with APC techniques such as MPC
- Real-time optimization of plant operating conditions
- Expansion of the natural language interface to cover more aspects of plant operation
- Inclusion of operational constraints 


## 🗣️ Implications for Plant Communication

This project demonstrates the potential for using AI to facilitate more intuitive and efficient communication between operators and chemical plants. By providing natural language explanations and suggestions, it paves the way for more accessible and user-friendly interfaces in industrial control rooms. 

A key innovation is the provision of detailed explanations for suggested control actions. Instead of simply outputting numerical PID gain values, the system offers context and reasoning behind its recommendations. This approach:

- 🧠 Enhances operator understanding of control decisions
- 🛠️ Builds trust in the AI-assisted control system
- 📚 Serves as an educational tool, helping operators learn about process dynamics
- 🔍 Allows for more informed decision-making by human operators

These explanatory features represent a significant step towards more transparent and interpretable AI systems in industrial control, potentially leading to improved safety, efficiency, and operator confidence in complex chemical processes.

## 📄 License

MIT License
