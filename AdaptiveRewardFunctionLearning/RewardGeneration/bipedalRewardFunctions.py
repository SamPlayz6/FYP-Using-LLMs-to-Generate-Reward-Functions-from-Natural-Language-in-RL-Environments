import numpy as np

def badRewardBipedal(observation, action):
    """
    A deliberately poor reward function for BipedalWalker that prioritizes
    the wrong aspects of locomotion.
    """
    # Extract relevant observations
    hull_angle = observation[0]  # Hull angle
    hull_vel_x = observation[1]  # Horizontal velocity
    
    # Bad reward - penalizes horizontal movement and rewards poor posture
    movement_penalty = -5.0 * abs(hull_vel_x)    # Penalize forward movement
    angle_reward = 0.1 * (1.0 - np.cos(hull_angle))  # Reward non-vertical posture
    
    return float(movement_penalty + angle_reward)

def stabilityRewardBipedal(observation, action):
    """
    Reward function focused on maintaining balance and good posture.
    """
    # Extract observations
    hull_angle = observation[0]  # Hull angle
    hull_angular_vel = observation[2]  # Hull angular velocity
    
    # Calculate rewards
    posture_reward = np.cos(hull_angle)  # Highest reward when upright
    stability_penalty = -0.1 * abs(hull_angular_vel)  # Penalize rapid rotation
    
    # Add joint position/velocity terms for smooth movement
    joint_positions = observation[4:8]  # 4 joint positions
    joint_velocities = observation[8:12]  # 4 joint velocities
    
    # Penalties for extreme joint positions/velocities
    joint_pos_penalty = -0.05 * sum([abs(pos) for pos in joint_positions])
    joint_vel_penalty = -0.05 * sum([abs(vel) for vel in joint_velocities])
    
    return float(posture_reward + stability_penalty + joint_pos_penalty + joint_vel_penalty)

def efficiencyRewardBipedal(observation, action):
    """
    Reward function focused on efficient movement and energy conservation.
    """
    # Extract observations
    hull_vel_x = observation[1]  # Horizontal velocity
    
    # Calculate rewards
    forward_reward = 1.0 * hull_vel_x  # Reward forward movement
    
    # Penalize inefficient energy use (high action values)
    # Note: For discretized actions, this may be less effective
    energy_penalty = -0.1 * sum([a**2 for a in action]) if hasattr(action, "__iter__") else -0.1 * abs(action)
    
    # Reward for leg contact with ground (indicates stable stance)
    leg_contacts = observation[12:14]  # Contact indicators
    contact_reward = 0.1 * sum(leg_contacts)
    
    return float(forward_reward + energy_penalty + contact_reward)

def potentialBasedRewardBipedal(observation, action):
    """Potential-based reward shaping for BipedalWalker"""
    # Extract key state variables
    hull_angle = observation[0]
    hull_vel_x = observation[1]
    gamma = 0.99
    
    def potential(obs):
        # Potential function based on posture and movement
        posture_potential = -abs(obs[0])  # Hull angle from vertical
        velocity_potential = obs[1]  # Forward velocity
        joint_potential = -sum([abs(pos) for pos in obs[4:8]])  # Joint positions
        
        return posture_potential + 2*velocity_potential + 0.5*joint_potential
    
    # Estimate next state (simple approximation)
    next_obs = observation.copy()
    next_obs[0] += observation[2] * 0.1  # Hull angle + angular velocity * time step
    next_obs[1] += observation[3] * 0.1  # Horizontal velocity + acceleration * time step
    
    # PBRS formula: γΦ(s') - Φ(s)
    current_potential = potential(observation)
    next_potential = potential(next_obs)
    shaped_reward = gamma * next_potential - current_potential
    
    return 1.0 + shaped_reward

def energyBasedRewardBipedal(observation, action):
    """Energy-based reward for BipedalWalker"""
    # Extract observations
    hull_angle = observation[0]
    hull_vel_x = observation[1]
    hull_vel_y = observation[3]
    joint_positions = observation[4:8]
    joint_velocities = observation[8:12]
    
    # Kinetic energy components
    linear_ke = 0.5 * (hull_vel_x**2 + hull_vel_y**2)
    rotational_ke = 0.5 * observation[2]**2  # Angular velocity squared
    joint_ke = 0.5 * sum([v**2 for v in joint_velocities])
    
    # Potential energy components
    height = observation[3]  # Vertical position (approximate)
    gravitational_pe = 9.8 * height
    joint_pe = sum([abs(p) for p in joint_positions])  # Joint potential energy
    
    # Total energy
    total_energy = linear_ke + rotational_ke + joint_ke + gravitational_pe + joint_pe
    
    # Reward is negative energy (we want to minimize) plus forward velocity incentive
    reward = -0.1 * total_energy + hull_vel_x
    
    return float(reward)

def baselineRewardBipedal(observation, action):
    """Standard baseline reward - always returns 1.0"""
    return 1.0