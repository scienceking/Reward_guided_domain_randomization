import numpy as np
from gymnasium import Env, spaces
from ray.rllib.env import EnvContext
import carla
import torch
import os
from collections import deque

os.environ["RAY_DEDUP_LOGS"] = "0"  
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

import random


class CustomEnv(Env):
    def __init__(self, config: EnvContext):
        super().__init__()
        #: [d, v, v_lead, a_lead]
        self.client = carla.Client("localhost", 2100)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town03')
        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter("vehicle.tesla.model3")[0]
        self.actor_list = []
        self.location_ego = carla.Location(x=120, y=130, z=1.8)
        self.rotation_ego = carla.Rotation(pitch=0, yaw=180, roll=0)
        self.transform_ego = carla.Transform(self.location_ego, self.rotation_ego)
        self.vehicle_ego = self.world.spawn_actor(self.model_3, self.transform_ego)
        self.actor_list.append(self.vehicle_ego)

        self.location_lead = carla.Location(x=110, y=130, z=1.8)
        self.rotation_lead = carla.Rotation(pitch=0, yaw=180, roll=0)
        self.transform_lead = carla.Transform(self.location_lead, self.rotation_lead)
        self.vehicle_lead = self.world.spawn_actor(self.model_3, self.transform_lead)
        self.actor_list.append(self.vehicle_lead)

        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
        self.state = None
        self.current_step = 0
        self.max_episode_steps = 40

        self.E_values = deque(maxlen=1500)  

        self.E1 = random.uniform(0.5, 2)
        self.E2 = random.uniform(1200, 2000)
        physics_ego = self.vehicle_ego.get_physics_control()
        physics_ego.mass =  self.E2
        self.vehicle_ego.apply_physics_control(physics_ego)
        print(f"[INFO] Ego mass set to: {physics_ego.mass} kg")

        self.set_fixed_time_step()
        self.friction()

        self.episode_return = 0 
        self.time_count =0

        self.segment_counts = {  
            ((0.5, 1), (1200, 1400)): 0, ((0.5, 1), (1400, 1600)): 0, ((0.5, 1), (1600, 1800)): 0, ((0.5, 1), (1800, 2000)): 0,
            ((1, 1.5), (1200, 1400)): 0, ((1, 1.5), (1400, 1600)): 0, ((1, 1.5), (1600, 1800)): 0, ((1, 1.5), (1800, 2000)): 0,
            ((1.5, 2), (1200, 1400)): 0, ((1.5, 2), (1400, 1600)): 0, ((1.5, 2), (1600, 1800)): 0, ((1.5, 2), (1800, 2000)): 0
        }

    def remove_friction_triggers(self):
        self.world.tick()

        friction_actors = self.world.get_actors().filter('static.trigger.friction')
        for friction_actor in friction_actors:
            friction_actor.destroy()

        self.world.tick()

    def uniform_sampling(self):
        self.E1 = random.uniform(0.5, 2)
        self.E2 = random.uniform(1200, 2000)

    def update_segment_count(self, E1, E2):

        if 0.5 <= E1 < 1:
            E1_range = (0.5, 1)
        elif 1 <= E1 < 1.5:
            E1_range = (1, 1.5)
        else:
            E1_range = (1.5, 2)

        if 1200 <= E2 < 1400:
            E2_range = (1200, 1400)
        elif 1400 <= E2 < 1600:
            E2_range = (1400, 1600)
        elif 1600 <= E2 < 1800:
            E2_range = (1600, 1800)
        else:
            E2_range = (1800, 2000)

        segment_key = (E1_range, E2_range)
        self.segment_counts[segment_key] += 1
        # print(f"Updated segment {segment_key}: {self.segment_counts[segment_key]}")



    def reward_guided_sampling(self, temperature=1.8):
        weights = [abs(episode_return) for _, episode_return in self.E_values]
        weights = [(weight + 1e-6) ** temperature for weight in weights]  # 应用温度参数
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]



        chosen_index = random.choices(range(len(self.E_values)), weights=probabilities, k=1)[0]
        chosen_E = self.E_values[chosen_index]
        # print(chosen_E)
        # print(f"Chosen E: {chosen_E}")


        del self.E_values[chosen_index]


        self.E1, self.E2 = chosen_E[0]

        self.E1 += np.random.normal(0, 0.02)
        self.E2 += np.random.normal(0, 10)
        print(f"Updated E1 with noise: {self.E1}, Updated E2 with noise: {self.E2}")




    def environment_sampling(self):
        if len(self.E_values) < 300:
            self.uniform_sampling()
        else:   
            p = max(0.1, np.exp(-self.time_count / 2000))
            if random.random() < p:
                self.uniform_sampling()
            else:
                self.reward_guided_sampling()
        self.update_segment_count(self.E1, self.E2)
        

    def reset(self, *, seed=None, options=None):
        self.time_count += 1
        if self.time_count % 100 == 0:
            with open('segment_counts.txt', 'w') as txt_file:
                for key, value in self.segment_counts.items():
                    txt_file.write(f"{key}: {value}\n")

        # print(self.time_count)
        self.environment_sampling()
        # print(self.segment_counts)
        self.friction()
        super().reset(seed=seed)
        d_0_ego = random.uniform(110, 120)
        new_location_ego = carla.Location(x=d_0_ego, y=130, z=1.8)
        new_rotation_ego = carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000)
        new_transform_ego = carla.Transform(new_location_ego, new_rotation_ego)
        v_0_ego = random.uniform(5.5, 9.5)
        self.vehicle_ego.set_target_velocity(carla.Vector3D(x=-v_0_ego, y=0, z=0))
        self.vehicle_ego.set_transform(new_transform_ego)

        d_0_lead = d_0_ego - random.uniform(10, 12)
        new_location_lead = carla.Location(x=d_0_lead, y=130, z=1.8)
        new_rotation_lead = carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000)
        new_transform_lead = carla.Transform(new_location_lead, new_rotation_lead)
        v_0_lead = random.uniform(v_0_ego, 10)
        self.vehicle_lead.set_target_velocity(carla.Vector3D(x=-v_0_lead, y=0, z=0))
        self.vehicle_lead.set_transform(new_transform_lead)
        for i in range(3):
            self.world.tick()
        self.state = np.array([
            self.vehicle_ego.get_transform().location.x,  # d
            self.vehicle_ego.get_velocity().x,  # v
            self.vehicle_lead.get_velocity().x,  # v_lead
            np.random.uniform(-3, 3)  # a_lead
        ], dtype=np.float32)
        self.current_step = 0  # Reset step count
        self.episode_return = 0  #  episode return
        return self.state, {}

    def step(self, action):
        # print(f"Step {self.current_step + 1}: Current state: {self.state}, Action: {action}")  # Print current state and action
        # Compute next state and reward
        self.control_vehicle(action)
        for i in range(20):
            self.world.tick()

        next_state = self.sensor()
        reward = self._get_reward(next_state, action)

        
        self.episode_return += reward

        # Check done conditions
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        done = truncated

        if done:
            # print(f"Episode finished. Total return: {self.episode_return}")
            self.E_values.append(((self.E1, self.E2), self.episode_return))

        # print(f"Step {self.current_step}: Next state: {next_state}, Reward: {reward}, Done: {done}, Truncated: {truncated}")  # Print next state and other details

        self.state = next_state
        return next_state, reward, done, truncated, {}

    def _get_reward(self, next_state, action):
        d_safe = 10
        v_max = 10
        v_min = 5

        
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().detach().numpy()

        next_d, next_v, next_v_lead, next_a_lead = next_state  # 直接解包

        R_danger = -1 * (d_safe - next_d) * self._indicator_function(next_d < d_safe)
        R_smooth = -(0.3) * abs(action)
        R_efficiency = -(0.5) * (self._indicator_function(abs(next_v) > v_max) * abs(abs(next_v) - v_max) +
                                           self._indicator_function(abs(next_v) < v_min) * abs(abs(next_v) - v_min))
        return float(R_danger + R_smooth + R_efficiency)

    def sensor(self):
        self.d = self.vehicle_ego.get_transform().location.x - self.vehicle_lead.get_transform().location.x
        self.v_ego = self.vehicle_ego.get_velocity().x
        self.v_lead = self.vehicle_lead.get_velocity().x
        next_state = np.array([self.d, self.v_ego, self.v_lead, self.a_lead], dtype=np.float32)
        return next_state

    def set_fixed_time_step(self, time_step=0.05):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = time_step
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

    def friction(self):
        self.remove_friction_triggers()
        friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
        extent = carla.Location(25000.0, 25000.0, 7000.0)
        friction_bp.set_attribute('friction', str(self.E1))
        friction_bp.set_attribute('extent_x', str(extent.x))
        friction_bp.set_attribute('extent_y', str(extent.y))
        friction_bp.set_attribute('extent_z', str(extent.z))
        transform = carla.Transform()
        transform.location = carla.Location(100.0, 0.0, 0.0)
        self.world.spawn_actor(friction_bp, transform)


        self.world.debug.draw_box(
            box=carla.BoundingBox(transform.location, extent * 1e-2),
            rotation=transform.rotation,
            life_time=1000,
            thickness=0.5,
            color=carla.Color(r=255, g=0, b=0)
        )

       
        physics_ego = self.vehicle_ego.get_physics_control()
        physics_ego.mass =  self.E2
        self.vehicle_ego.apply_physics_control(physics_ego)
        


    def control_vehicle(self, action):
        ego_yaw = self.vehicle_ego.get_transform().rotation.yaw
        target_yaw = 180
        if ego_yaw < 0:
            current_yaw = 360 + ego_yaw
        else:
            current_yaw = ego_yaw
        yaw_error = target_yaw - current_yaw

        steer_ego = yaw_error * 0.05
        steer_ego = 0.1*np.clip(steer_ego, -1.0, 1.0)  #


        action = 0.5 * action[0]

        if action < 0:
            self.vehicle_ego.apply_control(carla.VehicleControl(throttle=0.0, brake=abs(action), steer=steer_ego))
        else:
            self.vehicle_ego.apply_control(carla.VehicleControl(throttle=action, brake=0, steer=steer_ego))

        ##################### lead ##########################
        lead_yaw = self.vehicle_lead.get_transform().rotation.yaw
        if lead_yaw < 0:
            current_lead_yaw = 360 + lead_yaw
        else:
            current_lead_yaw = lead_yaw
        yaw_lead_error = target_yaw - current_lead_yaw

        steer_lead = yaw_lead_error * 0.05
        steer_lead = 0.1*np.clip(steer_lead, -1.0, 1.0)  #

        v_desired = 5
        self.v_lead = self.vehicle_lead.get_velocity().x
        self.a_lead = 0.5 * (v_desired - abs(self.v_lead))
        if self.a_lead > 0:
            self.vehicle_lead.apply_control(carla.VehicleControl(throttle=self.a_lead, brake=0.0, steer=steer_lead))
        else:
            self.vehicle_lead.apply_control(carla.VehicleControl(throttle=0, brake=abs(self.a_lead), steer=steer_lead))

    @staticmethod
    def _indicator_function(condition):
        return 1 if condition else 0

    def print_segment_counts(self):
        print("Segment counts:")
        for segment, count in self.segment_counts.items():
            print(f"Segment {segment}: {count}")
