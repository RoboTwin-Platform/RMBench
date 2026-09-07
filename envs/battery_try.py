from ._base_task import Base_Task
from .utils import *

class battery_try(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.battery_slot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="017_battery_slot_gauge",
            modelid=10127,
            xlim=[0.0, 0.0],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
            qpos=[0.707, 0 , 0, 0.707],
            fix_root_link=True,
        )
        self.battery_slot_base_name = "battery_slot_base"
        for link in self.battery_slot.actor.get_links():
            if link.get_name() == "base_link":
                link.entity.set_name(self.battery_slot_base_name)
                break
        self.set_dashboard_off()

        self.battery1 = rand_create_battery(
            scene=self,
            xlim=[-0.2, -0.25],
            ylim=[-0.2, -0.1],
            name="battery1",
        )
        self.battery2 = rand_create_battery(
            scene=self,
            xlim=[0.2, 0.25],
            ylim=[-0.2, -0.1],
            name="battery2",
        )
        self.target_pose1_p = self.battery_slot.get_pose().p + np.array([-0.02, 0.0, 0.0])
        self.target_pose2_p = self.battery_slot.get_pose().p + np.array([0.02, 0.0, 0.0])   
        self.quat_of_target_pose_0 = np.array([0.707, -0.707, 0.0, 0.0], dtype=np.float64)
        self.quat_of_target_pose_1 = np.array([0.0, 0.0, 0.707, -0.707], dtype=np.float64)
        self.combination_lst = [[0,0], [1,0], [1,1], [0,1]]
        self.correct_combination = self.combination_lst[np.random.randint(1, len(self.combination_lst))]
    
    def set_dashboard_off(self, joint_names=['needle_joint'], target=0.0): 
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot   
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]    
        for joint_name in joint_names:
            idx = joint_names_list.index(joint_name)
            qpos = art.get_qpos()
            qpos[idx] = target   
            art.set_qpos(qpos) 
            joints[idx].set_drive_target(target)

    def set_dashboard_on(self, joint_names=['needle_joint'], target=1.57): 
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot   
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]    
        for joint_name in joint_names:
            idx = joint_names_list.index(joint_name)
            qpos = art.get_qpos()
            qpos[idx] = target   
            art.set_qpos(qpos) 
            joints[idx].set_drive_target(target)

    def check_dashboard_on(self):
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot  
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]  
        idx =  joint_names_list.index('needle_joint')
        qpos = art.get_qpos()
        if qpos[idx] > 1.4:
            return True
        else:
            return False

    def get_battery_state(self, battery_actor):
        # Local +Z is the positive cap. Target poses map it to +Y (state 0) or -Y (state 1).
        axis = battery_actor.get_pose().to_transformation_matrix()[:3, 2]
        return 0 if axis[1] >= 0.0 else 1
        
    def check_battery_in_slot(self, battery_actor, target_pose_p):
        curr_p = battery_actor.get_pose().p
        battery_in_slot = (np.linalg.norm(curr_p[:2] - target_pose_p[:2])<0.02)
        return battery_in_slot
    
    def get_curr_combination(self):
        battery1_state = self.get_battery_state(self.battery1)
        battery2_state = self.get_battery_state(self.battery2)
        curr_combination = [battery1_state, battery2_state]
        return curr_combination

    def place_for_battery1(self, arm, state):
        curr_state = self.get_battery_state(self.battery1)
        if (state == 0 and state != curr_state):
            self.move(self.grasp_actor(self.battery1, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03), language_annotation="Pick up the left battery and place it into the battery slot in the positive direction.")
            self.move(self.move_by_displacement(arm_tag=arm, z=0.05), language_annotation="Pick up the left battery and place it into the battery slot in the positive direction.")
            self.move(self.place_actor(self.battery1, arm_tag=arm, target_pose=self.target_pose1_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                      language_annotation="Pick up the left battery and place it into the battery slot in the positive direction.")
            self.check_correct_combination()
            self.move(self.back_to_origin(arm_tag=arm), language_annotation="Pick up the left battery and place it into the battery slot in the positive direction.")
        elif (state == 1 and state != curr_state):
            self.move(self.grasp_actor(self.battery1, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03), language_annotation="Pick up the left battery and place it into the battery slot in the negative direction.")
            self.move(self.move_by_displacement(arm_tag=arm, z=0.05), language_annotation="Pick up the left battery and place it into the battery slot in the negative direction.")
            self.move(self.place_actor(self.battery1, arm_tag=arm, target_pose=self.target_pose1_p.tolist()+self.quat_of_target_pose_1.tolist(), functional_point_id=0, constrain="align"),
                      language_annotation="Pick up the left battery and place it into the battery slot in the negative direction.")
            self.check_correct_combination()
            self.move(self.back_to_origin(arm_tag=arm), language_annotation="Pick up the left battery and place it into the battery slot in the negative direction.")

    def place_for_battery2(self, arm, state):
        curr_state = self.get_battery_state(self.battery2)
        if (state == 0 and state != curr_state):
            self.move(self.grasp_actor(self.battery2, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03), language_annotation="Pick up the right battery and place it into the battery slot in the positive direction.")
            self.move(self.move_by_displacement(arm_tag=arm, z=0.05), language_annotation="Pick up the right battery and place it into the battery slot in the positive direction.")
            self.move(self.place_actor(self.battery2, arm_tag=arm, target_pose=self.target_pose2_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                      language_annotation="Pick up the right battery and place it into the battery slot in the positive direction.")
            self.check_correct_combination()
            self.move(self.back_to_origin(arm_tag=arm), language_annotation="Pick up the right battery and place it into the battery slot in the positive direction.")
        elif (state == 1 and state != curr_state):
            self.move(self.grasp_actor(self.battery2, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03), language_annotation="Pick up the right battery and place it into the battery slot in the negative direction.")
            self.move(self.move_by_displacement(arm_tag=arm, z=0.05), language_annotation="Pick up the right battery and place it into the battery slot in the negative direction.")
            self.move(self.place_actor(self.battery2, arm_tag=arm, target_pose=self.target_pose2_p.tolist()+self.quat_of_target_pose_1.tolist(), functional_point_id=0, constrain="align"),
                      language_annotation="Pick up the right battery and place it into the battery slot in the negative direction.")
            self.check_correct_combination()
            self.move(self.back_to_origin(arm_tag=arm), language_annotation="Pick up the right battery and place it into the battery slot in the negative direction.")

    def check_correct_combination(self):
        curr_combination = self.get_curr_combination()
        if curr_combination == self.correct_combination:
            self.set_dashboard_on()
 
    def play_once(self):
        for idx, next_comb in enumerate(self.combination_lst):
            arm_tag = ArmTag("left" if self.battery1.get_pose().p[0] < 0 else "right")
            if idx == 0:
                self.move(
                    self.grasp_actor(self.battery1, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.03),
                    self.grasp_actor(self.battery2, arm_tag=arm_tag.opposite, pre_grasp_dis=0.1, grasp_dis=0.03),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self.move(
                    self.move_by_displacement(arm_tag=arm_tag, z=0.1),
                    self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.1),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self.move(self.place_actor(self.battery1, arm_tag=arm_tag, target_pose=self.target_pose1_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                          language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self.move(self.place_actor(self.battery2, arm_tag=arm_tag.opposite, target_pose=self.target_pose2_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                          language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self.move(self.back_to_origin(arm_tag=arm_tag.opposite), language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
            else:
                self.place_for_battery1(arm_tag, next_comb[0])
                self.place_for_battery2(arm_tag.opposite, next_comb[1])
            curr_state = self.get_curr_combination()
            if curr_state == self.correct_combination:
                break
        self.info['info'] = {}
        return self.info

    def check_success(self):
        current_combination = self.get_curr_combination()
        if current_combination == self.correct_combination:
            self.set_dashboard_on()
        if self.check_battery_in_slot(self.battery1, self.target_pose1_p) and self.check_battery_in_slot(self.battery2, self.target_pose2_p) \
            and current_combination == self.correct_combination and self.check_dashboard_on() \
            and self.check_actors_contact(self.battery1.get_name(), self.battery_slot_base_name) \
            and self.check_actors_contact(self.battery2.get_name(), self.battery_slot_base_name) \
            and self.is_left_gripper_open() and self.is_right_gripper_open():
            return True
        else:
            return False