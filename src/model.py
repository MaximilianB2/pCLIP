import numpy as np
import casadi as ca

class ChemicalReactor:
    def __init__(self, initial_state, initial_inputs, dt=0.2):
        self.state = np.array(initial_state)
        self.inputs = np.array(initial_inputs)
        self.dt = dt
        self.time = 0
        
        # Create CasADi model
        self.reactor_ode = self.create_reactor_model()
        
        # PID gains (initial values)
        self.Kp = np.array([1.0, 1.0])
        self.Ki = np.array([0.1, 0.1])
        self.Kd = np.array([0.05, 0.05])
        
        # Error history
        self.error_sum = np.zeros(2)
        self.prev_error = np.zeros(2)
        
        # Setpoints (initial values set to current state)
        self.setpoints = np.array([self.state[1], self.state[4]])  # Cb and V setpoints
        
        # Performance metrics
        self.error_history = []

    def update(self):
        # Calculate current error
        error = self.setpoints - self.state[[1, 4]]  
        
        # Update inputs using PID controller
        self.inputs = self.PID_position(error)
        
        # Simulate reactor for one time step using RK4
        self.state = self.rk4_integrator(self.reactor_ode, self.state, self.inputs, self.dt)
        
        # Update time
        self.time += self.dt
        
        # Update error history
        self.error_sum += error * self.dt
        self.prev_error = error
        
        # Update performance metrics
        self.error_history.append(error)
        if len(self.error_history) > 600:  # Keep last 10 minutes (assuming 0.1s time step)
            self.error_history.pop(0)

    def set_pid_gains(self, gains):
        self.Kp = np.array(gains[0::3])
        self.Ki = np.array(gains[1::3])
        self.Kd = np.array(gains[2::3])

    def set_setpoints(self, new_setpoints):
        self.setpoints = np.array(new_setpoints)

    def get_state(self):
        return {
            'time': self.time,
            'Ca': float(self.state[0]),
            'Cb': float(self.state[1]),
            'Cc': float(self.state[2]),
            'T': float(self.state[3]),
            'V': float(self.state[4]),
            'Tc': float(self.inputs[0]),
            'Fin': float(self.inputs[1]),
            'Cb_setpoint': float(self.setpoints[0]),
            'V_setpoint': float(self.setpoints[1]),
        }
    
    def rk4_integrator(self, f, x0, u, dt):
        """
        Runge-Kutta 4th order integrator
        f: function describing the ODE
        x0: initial state
        u: control input
        dt: time step
        """
        k1 = f(x0, u)
        k2 = f(x0 + dt/2 * k1, u)
        k3 = f(x0 + dt/2 * k2, u)
        k4 = f(x0 + dt * k3, u)
        return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def create_reactor_model(self):
        # States
        Ca = ca.SX.sym('Ca')
        Cb = ca.SX.sym('Cb')
        Cc = ca.SX.sym('Cc')
        T = ca.SX.sym('T')
        V = ca.SX.sym('V')
        x = ca.vertcat(Ca, Cb, Cc, T, V)

        # Controls
        Tc = ca.SX.sym('Tc')
        Fin = ca.SX.sym('Fin')
        u = ca.vertcat(Tc, Fin)

        # Parameters
        Tf = 350
        Caf = 1
        Fout = 100
        rho = 1000
        Cp = 0.239
        UA = 5e4
        mdelH_AB = 5e3
        EoverR_AB = 8750
        k0_AB = 7.2e10
        mdelH_BC = 4e3
        EoverR_BC = 10750
        k0_BC = 8.2e10

        # Reaction rates
        rA = k0_AB * ca.exp(-EoverR_AB / T) * Ca
        rB = k0_BC * ca.exp(-EoverR_BC / T) * Cb

        # ODEs
        dCadt = (Fin * Caf - Fout * Ca) / V - rA
        dCbdt = rA - rB - Fout * Cb / V
        dCcdt = rB - Fout * Cc / V
        dTdt = (Fin / V * (Tf - T) + 
                mdelH_AB / (rho * Cp) * rA + 
                mdelH_BC / (rho * Cp) * rB + 
                UA / V / rho / Cp * (Tc - T))
        dVdt = Fin - Fout

        xdot = ca.vertcat(dCadt, dCbdt, dCcdt, dTdt, dVdt)

        return ca.Function('reactor_ode', [x, u], [xdot])
    
    def PID_position(self, error):
        """
        Position form PID controller
        """
        derivative = (error - self.prev_error) / self.dt

        # Calculate PID output
        Tc =320+ (self.Kp[0] * error[0] + 
              self.Ki[0] * self.error_sum[0] + 
              self.Kd[0] * derivative[0])
        F = 100+ (self.Kp[1] * error[1] + 
             self.Ki[1] * self.error_sum[1] + 
             self.Kd[1] * derivative[1])

        # Clamp outputs between operational limits
        Tc = min(max(Tc, 290.), 390.)
        F = min(max(F, 99.), 102.)

        return np.array([float(Tc), float(F)], dtype=np.float64)
    
    def run_simulation(self, duration):
        steps = int(duration / self.dt)
        history = []
        for _ in range(steps):
            self.update()
            history.append(self.get_state())
        return history