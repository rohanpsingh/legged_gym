from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

PI = 3.141592653589793

class HRP5PCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        #num_envs = 6144
        num_envs = 4096
        #num_envs = 2048
        num_observations = 83
        num_actions = 12
        episode_length_s = 10 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.80] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'RCY': 0,
            'RCR': 0.,
            'RCP': -28 * PI/180,
            'RKP': 50 * PI/180,
            'RAP': -22 * PI/180,
            'RAR': 0,

            'LCY': 0,
            'LCR': 0.,
            'LCP': -28 * PI/180,
            'LKP': 50 * PI/180,
            'LAP': -22 * PI/180,
            'LAR': 0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        '''
        # PD Drive parameters:
        stiffness = {
            'RCY': 50.0, 'RCR': 50.0, 'RCP' : 150.0, 'RKP' : 200.0, 'RAP' : 40.0, 'RAR' : 40.0,
            'LCY': 50.0, 'LCR': 50.0, 'LCP' : 150.0, 'LKP' : 200.0, 'LAP' : 40.0, 'LAR' : 40.0,
        }  # [N*m/rad]
        damping = {
            'RCY': 5.0, 'RCR': 5.0, 'RCP' : 15.0, 'RKP' : 20.0, 'RAP' : 4.0, 'RAR' : 4.0,
            'LCY': 5.0, 'LCR': 5.0, 'LCP' : 15.0, 'LKP' : 20.0, 'LAP' : 4.0, 'LAR' : 4.0,
        }  # [N*m*s/rad]     # [N*m*s/rad]
        '''

        # PD Drive parameters:
        stiffness = {
            'RCY': 200, 'RCR': 150, 'RCP' : 200, 'RKP' : 150, 'RAP' : 80, 'RAR' : 80,
            'LCY': 200, 'LCR': 150, 'LCP' : 200, 'LKP' : 150, 'LAP' : 80, 'LAR' : 80,
        }  # [N*m/rad]
        damping = {
            'RCY': 20, 'RCR': 15, 'RCP' : 20, 'RKP' : 15, 'RAP' : 8, 'RAR' : 8,
            'LCY': 20, 'LCR': 15, 'LCP' : 20, 'LKP' : 15, 'LAP' : 8, 'LAR' : 8,
        }  # [N*m*s/rad]     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 1000. # time before command are changed[s]
        class ranges:
            lin_vel_x = [0., 0.4] # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hrp5p/mjcf/HRP5Pmain.xml'
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hrp5p/urdf/HRP5Pmain.urdf'
        name = "hrp5p"
        foot_name = 'leg_Link5'
        terminate_after_contacts_on = ['Body', 'leg_Link0', 'leg_Link1', 'leg_Link2', 'leg_Link3', 'leg_Link4']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

        density = 0.001
        angular_damping = 0
        linear_damping = 0
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards:
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1200.
        only_positive_rewards = False
        base_height_target = 0.79
        class scales:
            lin_vel_xy = 0.2
            ang_vel_z = 0.2
            orient = 0.05
            torque = 0.1
            base_height = 0.05
            action_rate = 0.05
            upperbody = 0.1
            posture = 0.1
            clock_frc = 0.2
            clock_vel = 0.2

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_torques = 0.000001
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HRP5PCfgPPO(LeggedRobotCfgPPO):
    
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'hrp5p'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
