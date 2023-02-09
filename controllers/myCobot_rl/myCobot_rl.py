"""myCobot_rl controller."""

import sys
from controller import Supervisor
import math
import os
import torch
from stable_baselines3.common.logger import configure
import time
from PIL import Image

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
    from stable_baselines3 import A2C
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import DictReplayBuffer
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')

class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1):
        super().__init__()

        # Open AI Gym generic
        joint_range = 5/6 * np.pi
        self.action_space = gym.spaces.Box(
            low=np.array(6*[-joint_range]),
            high=np.array(6*[joint_range]),
            dtype=np.float32
        )
        top_image_space = gym.spaces.Box(
            low = 0, high = 255, shape = (4, 240, 320), dtype=np.uint8
        )
        side_image_space = gym.spaces.Box(
            low = 0, high = 255, shape = (4, 320, 240), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {"Current joint Angles": self.action_space,
            "Top Image": top_image_space, "Side Image": side_image_space})

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())*4096
        self.__motors = []
        self.__cam_top = None
        self.__cam_ee = None
        self.step_count = 0
        self.__arm = self.getSelf()
        
    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        # self.restartController()
        super().step(self.__timestep)

        # Create the arm chain from the URDF
        base_path = '/home/jtgenova/Documents/GitHub/myCobot/'
        os.chdir(base_path)
        filename = "myCobot.urdf"
        # print(filename)
        self.__armChain = Chain.from_urdf_file(filename)
        for i in [0, 7]:
            self.__armChain.active_links_mask[i] = False

        # Initialize the arm motors and encoders.
        motors = []
        for link in self.__armChain.links:
            if 'motor' in link.name:
                motor = self.getDevice(link.name)
                motor.setVelocity(1.0)
                position_sensor = motor.getPositionSensor()
                position_sensor.enable(self.__timestep)
                motors.append(motor)
        self.__motors = motors
        
                
        # initialize cameras
        self.__cam_top = self.getDevice('camera_top')
        self.__cam_ee = self.getDevice('camera_side')
        self.__cam_top.enable(self.__timestep)
        self.__cam_ee.enable(self.__timestep)

        # getDef of objects
        self.target = self.getFromDef('waypoint')
        self.cube = self.getFromDef('cube')
        self.end_effector = self.getFromDef('ee_tool')
        self.robot = self.getFromDef('robot')
        
        # randomize starting position of robot
        robot_position = self.robot.getField('translation')
        robot_x = -0.005 + np.random.uniform(low=0, high=10)*0.001
        robot_y = -0.15 + np.random.uniform(low=0, high=5)*0.001
        robot_rand_position = [robot_x, robot_y, 0.02 ]
        robot_position.setSFVec3f(robot_rand_position)

        # randomize cube position
        random_x = -0.03 + np.random.uniform(low=0, high=8)*0.01
        random_y = -0.03 + np.random.uniform(low=0, high=8)*0.01
        rand_damage_position = [random_x, random_y, 0.02]
        translation_cube_field = self.cube.getField('translation')
        translation_cube_field.setSFVec3f(rand_damage_position)

        # randomize starting position of cam_top
        cam_top_pos = self.getFromDef('cam_top_pose')
        cam_top_position = cam_top_pos.getField('translation')
        cam_top_x = -0.015 + np.random.uniform(low=0, high=30)*0.001
        cam_top_y = np.random.uniform(low=0, high=30)*0.001
        cam_top_z = 0 - np.random.uniform(low=0, high=10)*0.001
        cam_top_rand_position = [cam_top_x, cam_top_y, cam_top_z]
        cam_top_position.setSFVec3f(cam_top_rand_position)

        # randomize starting position of cam_side
        cam_side_pos = self.getFromDef('cam_side_pose')
        cam_side_position = cam_side_pos.getField('translation')
        cam_side_x = -0.01 + np.random.uniform(low=0, high=20)*0.001
        cam_side_y = -0.01 + np.random.uniform(low=0, high=20)*0.001
        cam_side_z = -0.01 + np.random.uniform(low=0, high=20)*0.001
        cam_side_rand_position = [cam_side_x, cam_side_y, cam_side_z]
        cam_side_position.setSFVec3f(cam_side_rand_position)
        
        # randomize starting position of joint1
        joint1_angle = -0.78 + np.random.uniform(low=0, high=157)*0.01
        self.__motors[0].setPosition(joint1_angle)

        # Open AI Gym generic
        # initialize joints

        # Internals
        super().step(self.__timestep)
        self.step_count = 0
        
        joint_angles = np.zeros(6)
        for i in range(len(self.__motors)):
            joint_angles[i] = self.__motors[i].getPositionSensor().getValue()
        
        top_array = np.frombuffer(self.__cam_top.getImage(), dtype=np.uint8).reshape(1, 4, 240, 320)
        side_array = np.frombuffer(self.__cam_ee.getImage(), dtype=np.uint8).reshape(1, 4, 320, 240)
        
        return dict({"Current joint Angles": joint_angles, "Top Image": top_array, "Side Image": side_array})

    def step(self, action):
        # Execute the 

        self.step_count += 1
        for i in range(len(self.__motors)):
            self.__motors[i].setPosition(float(action[i]))
        
        super().step(self.__timestep)

        # Observation
        targetPosition = self.target.getPosition()
        eePosition = self.end_effector.getPosition()
        
        targetOrientation = self.cube.getOrientation()
        eeOrientation = self.end_effector.getOrientation()
        orientation_z = targetOrientation[8] + eeOrientation[8]
        
        distance_x = targetPosition[0] - eePosition[0]
        distance_y = targetPosition[1] - eePosition[1]
        distance_z = targetPosition[2] - eePosition[2]
        total_distance = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        # manhattan_distance = abs(distance_x) + abs(distance_y) + abs(distance_z)

        done = False
        if (self.step_count >= 1) or (total_distance < 1e-5):
            # print('Steps: ' + str(self.step_count))
            print(f'Total Distance: {total_distance}, Orientation: {orientation_z}')
            done = True
        
        # reward
        r_distance = -total_distance
        # r_distance = -manhattan_distance
        r_orientation = -orientation_z
        reward = r_distance + r_orientation

        joint_angles = np.zeros(6)
        for i in range(len(self.__motors)):
            joint_angles[i] = self.__motors[i].getPositionSensor().getValue()
                
        top_array = np.frombuffer(self.__cam_top.getImage(), dtype=np.uint8).reshape(1, 4, 240, 320)
        side_array = np.frombuffer(self.__cam_ee.getImage(), dtype=np.uint8).reshape(1, 4, 320, 240)
        
        return dict({"Current joint Angles": joint_angles, "Top Image": top_array, "Side Image": side_array}), reward, done, {}

    def add_ikpy(self, obs):
        IKPY_MAX_ITERATIONS = 128
        
        for i in range(len(self.__motors)):
            self.__motors[i].setPosition(float(obs[i]))
            
        targetPosition = self.target.getPosition()
        eePosition = self.end_effector.getPosition()
        armPosition = self.__arm.getPosition()
        
        # Compute the position of the target relatively to the arm.
        x = targetPosition[0] - armPosition[0]
        y = targetPosition[1] - armPosition[1]
        z = targetPosition[2] - armPosition[2]
        
        # Compute the position of the target relatively to the end-effector.
        distance_x = targetPosition[0] - eePosition[0]
        distance_y = targetPosition[1] - eePosition[1]
        distance_z = targetPosition[2] - eePosition[2]
        total_distance = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        
        # Call "ikpy" to compute the inverse kinematics of the arm.
        initial_position = [0] + [m.getPositionSensor().getValue() for m in self.__motors] + [0]
        ikResults = self.__armChain.inverse_kinematics([x, y, z],[0, 0, -1],orientation_mode="Z", max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

        # Recalculate the inverse kinematics of the arm if necessary.
        position = self.__armChain.forward_kinematics(ikResults)
        squared_distance = (position[0, 3] - x)**2 + (position[1, 3] - y)**2 + (position[2, 3] - z)**2
        if math.sqrt(squared_distance) > 1e-5:
            ikResults = self.__armChain.inverse_kinematics([x, y, z])
        
        # np_array = np.array(ikResults[1:7])   
        # np_round = np.around(np_array, 5)
        # np_list = list(np_round)
        # print(f'IK Joint Angles: {np_list}') 
        # print(ikResults[1:7])
        
        # if total_distance > 0.00125:
        #     for i in range(len(self.__motors)):
        #         self.__motors[i].setPosition(ikResults[i + 1])
        
        return ikResults[1:7]

def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()

    # Train
    # model = PPO('MultiInputPolicy', env, n_steps=25, verbose=1, tensorboard_log="./log/")
    # model = A2C('MultiInputPolicy', env, n_steps=25, verbose=1, tensorboard_log="./log/")
    # model = TD3("MultiInputPolicy", env, buffer_size=100, replay_buffer_class=DictReplayBuffer, batch_size=25, verbose=1, tensorboard_log="./log/")
    # model = DDPG("MultiInputPolicy", env, buffer_size=250, replay_buffer_class=DictReplayBuffer, batch_size=50, verbose=1, tensorboard_log="./log/")
    
    # model.learn(total_timesteps=100,log_interval=25, tb_log_name='run_temp', reset_num_timesteps=False)
    
    # model hyperparameters
    buffer_size = 1000
    replay_buffer_class = DictReplayBuffer
    batch_size = 128
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    ent_coef = 'auto_0.1'
    
    itererations = 100
    episodes = 1000
    batch_inds = np.random.randint(0, 5, size=10)
    model = SAC("MultiInputPolicy", env, buffer_size=buffer_size, replay_buffer_class=replay_buffer_class,action_noise=action_noise, ent_coef=ent_coef, batch_size=batch_size, verbose=1, tensorboard_log="./data/")
    for iter in range(itererations):
        
        for ep in range(episodes):
            print("Episode #: " + str(ep+1))
            done = False
            obs = env.reset()
            obs_old = obs
            while done != True:
                actions = env.add_ikpy(obs_old['Current joint Angles'])
                obs_new, reward, done, infos = env.step(actions)
                model.replay_buffer.add(obs_old, obs_new, actions, reward, done, infos)
                obs_old = obs_new
                
        # buffer_obs, buffer_actions, buffer_next_obs, buffer_dones, buffer_reward = model.replay_buffer._get_samples(batch_inds)
        # print(f'Actions from replay buffer: {buffer_actions}')
    
        model.learn(total_timesteps=1e3,log_interval=100, tb_log_name=f'sac_{iter+1}k', reset_num_timesteps=False)
        model.save(f'models/sac_{iter+1}k')
        del model
        os.chdir('/home/jtgenova/Documents/GitHub/myCobot/') 
        model = SAC.load(f'models/sac_{iter+1}k', env)
                               
    # buffer_obs, buffer_actions, buffer_next_obs, buffer_dones, buffer_reward = model.replay_buffer._get_samples(batch_inds)
    # print(buffer_actions)
    
    # model.save('models/sac_100k')
    # env.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

    # add noise for exploration
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    
    # model = SAC("MultiInputPolicy", env, buffer_size=5000, replay_buffer_class=DictReplayBuffer, action_noise=action_noise, ent_coef='auto_0.1', batch_size=128, verbose=1, tensorboard_log="./log/")
    # model.learn(total_timesteps=20e3,log_interval=1000, tb_log_name='run_sac_new20k', reset_num_timesteps=False)
    # model.save('models/sac_new20k')

    # iterations = 5
    
    # for i in range(2, iterations+1):
    #     del model
    #     os.chdir('/home/jtgenova/Documents/GitHub/myCobot/') 
    #     model = SAC.load(f'models/sac_new{(i-1)*2}0k', env)
    #     model.learn(total_timesteps=20e3,log_interval=1000, tb_log_name=f'run_sac_new{2*i}0k', reset_num_timesteps=False)
    #     model.save(f'models/sac_new{2*i}0k')
    
    env.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

    obs = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)
        print(f'Actions from predict: {action[0].reshape(-1)}')
        obs, reward, done, info = env.step(action[0].reshape(-1))
        # env.render()
        if done:
            obs = env.reset()

if __name__ == '__main__':
    main()
    