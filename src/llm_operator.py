import matplotlib.pyplot as plt
from model import ChemicalReactor
import os
import json
from datetime import datetime
import requests
import re
class LLMOperatorInterface:
    def __init__(self):
        self.initial_state = [1, 0, 0, 295, 100]
        self.initial_inputs = [300, 100]
        self.setpoints = [0.8, 101]
        self.current_gains = [50,0, 0, 6, 0, 10]   # Initial unstable gains
        self.log_file = f"llm_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.interaction_history = []
        # Claude API settings
        self.api_key = os.environ.get('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not set")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
    def plot_results(self, history):
        time = [state['time'] for state in history]
        Cb = [state['Cb'] for state in history]
        V = [state['V'] for state in history]
        Cb_setpoint = [state['Cb_setpoint'] for state in history]
        V_setpoint = [state['V_setpoint'] for state in history]
        Tc = [state['Tc'] for state in history]
        Fin = [state['Fin'] for state in history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

        ax1.plot(time, Cb, label='Cb')
        ax1.plot(time, Cb_setpoint, 'r--', label='Cb Setpoint')
        ax1.set_ylabel('Concentration B (mol/m^3)')
        ax1.set_title('Concentration B vs Time')
        ax1.legend()

        ax2.plot(time, V, label='V')
        ax2.plot(time, V_setpoint, 'r--', label='V Setpoint')
        ax2.set_ylabel('Volume (m^3)')
        ax2.set_title('Volume vs Time')
        ax2.legend()

        ax3.step(time, Tc, where='post', label='Tc')
        ax3.set_ylabel('Cooling Temperature (K)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Cooling Temperature vs Time')
        ax3.legend()

        ax4.step(time, Fin, where='post', label='Fin')
        ax4.set_ylabel('Inlet Flow Rate (m^3/s)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Inlet Flow Rate vs Time')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def run_simulation(self, gains):
        reactor = ChemicalReactor(self.initial_state, self.initial_inputs)
        reactor.set_pid_gains(gains)
        reactor.set_setpoints(self.setpoints)
        history = reactor.run_simulation(duration=10)
        
        # Calculate performance metrics
        Cb_error = sum([abs(state['Cb'] - self.setpoints[0]) for state in history]) / len(history)
        V_error = sum([abs(state['V'] - self.setpoints[1]) for state in history]) / len(history)
        Cb_osc = (sum([(state['Cb'] - self.setpoints[0])**2 for state in history]) / len(history))**0.5
        V_osc = (sum([(state['V'] - self.setpoints[1])**2 for state in history]) / len(history))**0.5
        
        # Calculate control input oscillations
        Tc_oscillations = sum(1 for i in range(1, len(history)) if abs(history[i]['Tc'] - history[i-1]['Tc']) > 1e-6)
        Fin_oscillations = sum(1 for i in range(1, len(history)) if abs(history[i]['Fin'] - history[i-1]['Fin']) > 1e-6)
        
        # Check for control inputs hitting limits
        Tc_min, Tc_max = min(state['Tc'] for state in history), max(state['Tc'] for state in history)
        Fin_min, Fin_max = min(state['Fin'] for state in history), max(state['Fin'] for state in history)
        Tc_limit_hits = sum(1 for state in history if state['Tc'] in (Tc_min, Tc_max))
        Fin_limit_hits = sum(1 for state in history if state['Fin'] in (Fin_min, Fin_max))
        
        performance = {
            'Cb_error': Cb_error,
            'V_error': V_error,
            'Cb_osc': Cb_osc,
            'V_osc': V_osc,
            'Tc_oscillations': Tc_oscillations,
            'Fin_oscillations': Fin_oscillations,
            'Tc_limit_hits': Tc_limit_hits,
            'Fin_limit_hits': Fin_limit_hits,
            'Tc_range': (Tc_min, Tc_max),
            'Fin_range': (Fin_min, Fin_max)
        }
      
        return history, performance
    
    def get_llm_suggestion(self, history, performance):
        # Calculate metrics
        Cb_error = sum([abs(state['Cb'] - self.setpoints[0]) for state in history]) / len(history)
        V_error = sum([abs(state['V'] - self.setpoints[1]) for state in history]) / len(history)
        Cb_osc = (sum([(state['Cb'] - self.setpoints[0])**2 for state in history]) / len(history))**0.5
        V_osc = (sum([(state['V'] - self.setpoints[1])**2 for state in history]) / len(history))**0.5

        # Calculate trends
        Cb_trend = history[-1]['Cb'] - history[0]['Cb']
        V_trend = history[-1]['V'] - history[0]['V']
        # Calculate settling time (assuming 2% criterion)
        Cb_settling = next((i for i, state in enumerate(history) if abs(state['Cb'] - self.setpoints[0]) <= 0.02 * self.setpoints[0]), len(history))
        V_settling = next((i for i, state in enumerate(history) if abs(state['V'] - self.setpoints[1]) <= 0.02 * self.setpoints[1]), len(history))

        # Calculate oscillation frequency (zero crossings)
        Cb_zero_crossings = sum(1 for i in range(1, len(history)) if (history[i]['Cb'] - self.setpoints[0]) * (history[i-1]['Cb'] - self.setpoints[0]) < 0)
        V_zero_crossings = sum(1 for i in range(1, len(history)) if (history[i]['V'] - self.setpoints[1]) * (history[i-1]['V'] - self.setpoints[1]) < 0)

        # Simple stability metric (lower is more stable)
        stability_metric = (Cb_osc / self.setpoints[0] + V_osc / self.setpoints[1]) * (Cb_zero_crossings + V_zero_crossings)
        Cb_max = max(state['Cb'] for state in history)
        V_max = max(state['V'] for state in history)
        Cb_overshoot = max(0, (Cb_max - self.setpoints[0]) / self.setpoints[0] * 100)
        V_overshoot = max(0, (V_max - self.setpoints[1]) / self.setpoints[1] * 100)

        cb_performance = {
        'error': Cb_error,
        'oscillation': Cb_osc,
        'zero_crossings': Cb_zero_crossings,
        'trend': Cb_trend,
        'settling_time': Cb_settling,
        'overshoot': Cb_overshoot
        }
    
        v_performance = {
            'error': V_error,
            'oscillation': V_osc,
            'zero_crossings': V_zero_crossings,
            'trend': V_trend,
            'settling_time': V_settling,
            'overshoot': V_overshoot
        }

        interaction_history_str = "\n".join([
            f"Interaction {i+1}: Gains {interaction['gains']}, "
            f"Cb_error: {interaction['performance']['Cb_error']:.4f}, "
            f"V_error: {interaction['performance']['V_error']:.4f}, "
            f"Cb_osc: {interaction['performance']['Cb_osc']:.4f}, "
            f"V_osc: {interaction['performance']['V_osc']:.4f}"
            for i, interaction in enumerate(self.interaction_history[-5:])
        ])


  


        prompt = f"""PID expert: Optimize gains for chemical reactor with two pid controllers (Cb and V).

                Current gains [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]: {self.current_gains}
                Previous gains: {self.previous_gains if hasattr(self, 'previous_gains') else 'None'}

                Interaction history (last 5):
                {interaction_history_str}

                Final state:
                Cb: {history[-1]['Cb']:.2f}, V: {history[-1]['V']:.2f}, Tc: {history[-1]['Tc']:.2f}, Fin: {history[-1]['Fin']:.6f}

                Setpoints: Cb={self.setpoints[0]}, V={self.setpoints[1]}

                Performance metrics for Cb controller:
                1. Avg error: {performance['Cb_error']:.4f}
                2. Oscillation amplitude: {performance['Cb_osc']:.4f}
                3. Oscillation frequency (zero crossings): {cb_performance['zero_crossings']}
                4. Trend (start to end): {cb_performance['trend']:.4f}
                5. Settling time (steps): {cb_performance['settling_time']}
                6. Overshoot (%): {cb_performance['overshoot']:.2f}
                7. Control input (Tc) oscillations: {performance['Tc_oscillations']}
                8. Control input (Tc) limit hits: {performance['Tc_limit_hits']}

                Performance metrics for V controller:
                1. Avg error: {performance['V_error']:.4f}
                2. Oscillation amplitude: {performance['V_osc']:.4f}
                3. Oscillation frequency (zero crossings): {v_performance['zero_crossings']}
                4. Trend (start to end): {v_performance['trend']:.4f}
                5. Settling time (steps): {v_performance['settling_time']}
                6. Overshoot (%): {v_performance['overshoot']:.2f}
                7. Control input (Fin) oscillations: {performance['Fin_oscillations']}
                8. Control input (Fin) limit hits: {performance['Fin_limit_hits']}




                Constraints and guidelines:
                - Gain values must be non-negative
                TAKE GREAT CARE TO KEEP GAINS WITHIN THEIR BOUNDS
                - Minimum allowable gains: Kp_Cb >= 25, Ki_Cb >= 0, Kd_Cb >= 0, Kp_V >= 0, Ki_V >= 0.01, Kd_V >= 0.01
                - Maximum allowable gains: Kp_Cb <= 125, Ki_Cb <= 75, Kd_Cb <= 1, Kp_V <= 6, Ki_V <= 10, Kd_V <= 10
                - Aim for a stable but responsive controller for each controller independently
                - Consider both increasing and decreasing gains
                - Adjust gains for each controller (Cb and V) separately based on their individual performance
                - It is not necessary to change all gains every time. If a controller is performing well, its gains can be left unchanged
                
                Focus areas:
                1. Reduce oscillations (both amplitude and frequency) for each controller
                2. Minimize overshoot for each controller
                3. Improve settling time without sacrificing stability
                4. Reduce steady-state error for each controller
                5. Minimize control input oscillations and limit hitting
                6. Keep gains within their bounds

                Important: 
                - High proportional gains can lead to increased oscillations and instability
                - High derivative gains can lead to increased oscillations
                - Low gains may result in slow response and large steady-state errors
                - A large steady-state error can be due to a sluggish controller caused by a low proportional gain. But if paired with an oscillatory control this may be due to a large kp
                - Consider increasing the proportional gain if there's a persistent steady-state error and the system is not oscillating
                - Integral action helps eliminate steady-state error, and help combat oscillations but not too much
                - A control input (Tc or Fin) bouncing between max and min is a strong indicator of oscillations
                - High numbers of control input oscillations or limit hits suggest the need to reduce gains
                - Consider the trade-offs between different types of gains (P, I, D) and their effects on system behavior
                - Analyze and adjust the Cb controller (first 3 gains) and V controller (last 3 gains) independently
                - If one controller is performing well, you can leave its gains unchanged

                Gain adjustment guidelines:
                - Proportional (P) gain: Increases speed of response, but too high can cause instability and control input oscillations
                - Integral (I) gain: Eliminates steady-state error, but too high can cause oscillations and control input limit hitting
                - Derivative (D) gain: Reduces sensitivity to noise, but too high can make the system noisy and sensitive to disturbances (only use if necessary, try reducing proportional gains first to remove oscillations)

                Suggest gains to optimize performance. Use this exact format:
                Suggested gains: [Kp_Cb, Ki_Cb, Kd_Cb, Kp_V, Ki_V, Kd_V]
                Explanation for Cb controller: Brief explanation of changes (or why no changes were made) and expected improvements
                Explanation for V controller: Brief explanation of changes (or why no changes were made) and expected improvements
                Confidence: [Low/Medium/High] with reason

                Hint: Balancing P, I, and D gains is crucial. P for quick response and reducing steady-state error, I for eliminating steady-state error, D for damping oscillations.
                Remember to consider each controller separately and adjust their gains based on their individual performance.
                It's perfectly acceptable to leave gains unchanged if a controller is performing well.
                Pay close attention to control input oscillations and limit hitting as indicators of instability.
                Try PI-only before using kd as well
                """

        try:
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            suggestion = response.json()['content'][0]['text']
            
            self.save_interaction(prompt, suggestion)
            return suggestion
        except Exception as e:
            return f"Error: LLM generation failed - {str(e)}"
    def save_interaction(self, prompt, suggestion):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "suggestion": suggestion
        }
        try:
            with open(self.log_file, 'a') as f:
                json.dump(interaction, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving interaction: {e}")
    def start(self):
        print("Welcome to the Chemical Reactor PID Tuning Interface with LLM assistance.")
        print(f"Current unstable gains: {self.current_gains}")
        
        while True:
            print("\nEnter new gain values (comma-separated), 'a' for AI suggestion, or 'q' to quit:")
            user_input = input()
            
            if user_input.lower() == 'q':
                break
            
            try:
                if user_input.lower() == 'a':
                    history, performance = self.run_simulation(self.current_gains)
                    llm_suggestion = self.get_llm_suggestion(history, performance)
                    print("LLM suggestion:")
                    print(llm_suggestion)
                    
                    suggested_gains = self.parse_llm_suggestion(llm_suggestion)
                    
                    if suggested_gains:
                        print("\nWould you like to apply these suggested gains? (y/n)")
                        if input().lower() == 'y':
                            self.current_gains = suggested_gains
                            history, performance = self.run_simulation(self.current_gains)
                            self.plot_results(history)
                            print(f"Simulation completed with LLM suggested gains: {self.current_gains}")
                            
                            # Add to interaction history
                            self.interaction_history.append({
                                'gains': self.current_gains,
                                'performance': performance
                            })
                    continue

                new_gains = [float(g.strip()) for g in user_input.split(',')]
                if len(new_gains) != 6:
                    raise ValueError("Please enter exactly 6 gain values.")
                
                history, performance = self.run_simulation(new_gains)
                self.plot_results(history)
                print(f"Simulation completed with gains: {new_gains}")
                self.current_gains = new_gains

                # Add to interaction history
                self.interaction_history.append({
                    'gains': self.current_gains,
                    'performance': performance
                })

                print("Would you like the LLM's opinion on these gains? (y/n)")
                if input().lower() == 'y':
                    llm_opinion = self.get_llm_suggestion(history, performance)
                    print("LLM opinion:")
                    print(llm_opinion)

            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    def parse_llm_suggestion(self, suggestion):
        try:
            # Look for "Suggested gains:" or just "gains:" to be more flexible
            match = re.search(r'(?:Suggested )?gains: \[([\d\., ]+)\]', suggestion, re.IGNORECASE)
            
            if match:
                gains_str = match.group(1)
                suggested_gains = [float(g.strip()) for g in gains_str.split(',')]
                
                # Extract explanations for each controller
                cb_explanation = re.search(r'Explanation for Cb controller: (.+)', suggestion)
                v_explanation = re.search(r'Explanation for V controller: (.+)', suggestion)
                
                if cb_explanation and v_explanation:
                    print("Cb controller explanation:", cb_explanation.group(1))
                    print("V controller explanation:", v_explanation.group(1))
                
                return suggested_gains
            else:
                print("Couldn't parse gains from LLM response.")
                print("LLM response:", suggestion)  # Print the full response for debugging
                return None
        except Exception as e:
            print(f"Error parsing LLM suggestion: {e}")
            print("LLM response:", suggestion)  # Print the full response for debugging
            return None

if __name__ == "__main__":
    interface = LLMOperatorInterface()
    interface.start()
