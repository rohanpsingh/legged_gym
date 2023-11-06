import torch
import numpy as np

def create_simple_phase_reward(swing_duration, stance_duration, FREQ):
    # right-swing -> first-dblstance -> left-swing -> second-dblstance
    total_duration = swing_duration + stance_duration
    ss1 = [0  * FREQ, swing_duration  * FREQ]
    right_clock_func = [lambda x: torch.where((ss1[0] <= x) & (x < ss1[1]), 0, 1),
                        lambda x: torch.where((ss1[0] <= x) & (x < ss1[1]), 1, 0)]

    ss2 = [total_duration  * FREQ, (total_duration + swing_duration)  * FREQ]
    left_clock_func = [lambda x: torch.where((ss2[0] <= x) & (x < ss2[1]), 0, 1),
                       lambda x: torch.where((ss2[0] <= x) & (x < ss2[1]), 1, 0)]

    return [right_clock_func, left_clock_func]

def _calc_foot_frc_clock_reward(self, left_frc_fn, right_frc_fn):
    # constraints of foot forces based on clock
    max_foot_frc = torch.tensor(self.robot_mass*9.8, dtype=torch.float)

    left_frc_clock = left_frc_fn(self.phases)
    right_frc_clock = right_frc_fn(self.phases)

    desired_lfoot_frc = left_frc_clock.float()
    desired_lfoot_frc[right_frc_clock==1] = desired_lfoot_frc[right_frc_clock==1]*0.5
    left_frc_err = torch.square(desired_lfoot_frc - self.l_foot_frc/max_foot_frc)
    left_frc_score = torch.exp(-4*left_frc_err)

    desired_rfoot_frc = right_frc_clock.float()
    desired_rfoot_frc[left_frc_clock==1] = desired_rfoot_frc[left_frc_clock==1]*0.5
    right_frc_err = torch.square(desired_rfoot_frc - self.r_foot_frc/max_foot_frc)
    right_frc_score = torch.exp(-4*right_frc_err)

    foot_frc_score = (left_frc_score + right_frc_score)/2
    return foot_frc_score

def _calc_foot_vel_clock_reward(self, left_vel_fn, right_vel_fn):
    # constraints of foot velocities based on clock
    desired_max_foot_vel = torch.tensor(0.2, dtype=torch.float)

    left_vel_clock = left_vel_fn(self.phases)
    right_vel_clock = right_vel_fn(self.phases)

    left_vel_err = torch.square(left_vel_clock - self.l_foot_vel/desired_max_foot_vel)
    left_vel_score = torch.exp(-4*left_vel_err)

    right_vel_err = torch.square(right_vel_clock - self.r_foot_vel/desired_max_foot_vel)
    right_vel_score = torch.exp(-4*right_vel_err)

    foot_vel_score = (left_vel_score + right_vel_score)/2
    return foot_vel_score

