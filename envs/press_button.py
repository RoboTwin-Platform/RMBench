from  ._base_task import Base_Task
from .utils import *

class press_button(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        def create_card(pose, id):
            return create_actor(
                scene=self,
                pose=pose,
                modelname="004_numbercard",
                is_static=True,
                model_id=id,
                convex=True,
            )
        self.card_id_1 = np.random.randint(1, 10)
        self.card_id_2 = np.random.randint(1, 11-self.card_id_1)
        card_pose_1 = rand_pose(
            xlim=[-0.15, -0.15],
            ylim=[0.0, 0.0],
            qpos=[1, 0, 0, 0],
        )
        card_pose_2 = rand_pose(
            xlim=[0.02, 0.02],
            ylim=[0.0, 0.0],
            qpos=[1, 0, 0, 0],
        )
        self.card_1 = create_card(card_pose_1, self.card_id_1)
        self.card_2 = create_card(card_pose_2, self.card_id_2)

        self.button1 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.15, -0.15],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button1.set_mass(0.0001, ['button_cap'])
        self.set_button_unpressed(self.button1)

        self.button2 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.0, -0.0],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button2.set_mass(0.0001,['button_cap'])
        self.set_button_unpressed(self.button2)
        self.check_button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="006_check_button",
            modelid=10124,
            xlim=[0.15, 0.15],
            ylim=[-0.15, -0.15],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[0, 0, 0, 1],
            fix_root_link=True,
        )
        self.check_button.set_mass(0.0001,['button_cap'])
        self.set_button_unpressed(self.check_button)
        self.press_cnt_1 = 0 # left button press count
        self.press_cnt_2 = 0 # middle button press count
        self.press_cnt_check_button = 0 # check button press count
        self.press_flag_1 = False
        self.press_flag_2 = False
        self.press_flag_check_button = False

    def get_current_button_value(self, button_name, button_actor, joint_name="button_joint", target=0.0):
        if button_name == 'button':
            button_actor = button_actor
        else:
            button_actor = self.check_button
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor      
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx]

    def set_button_unpressed(self, button_actor, joint_name="button_joint", target=0.0): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        qpos[idx] = target   
        art.set_qpos(qpos) 
        joints[idx].set_drive_target(target)
    
    def check_button_pressed(self, button_actor, joint_name="button_joint", threshold=-0.005): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name) 
        qpos = art.get_qpos()
        if qpos[idx] < threshold:
            return True
        else:
            return False
    
    def update_button_reset(self, button_actor, flag_attr, joint_name="button_joint", threshold=-0.001):
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor
        joints = art.get_active_joints()
        joint_names = [j.get_name() for j in joints]
        idx = joint_names.index(joint_name)

        qpos = art.get_qpos()
        if qpos[idx] > threshold:
            setattr(self, flag_attr, False)

    def update_press_success(self, button_actor, flag_attr, cnt_attr):
        if self.check_button_pressed(button_actor) and not getattr(self, flag_attr):
            setattr(self, flag_attr, True)
            setattr(self, cnt_attr, getattr(self, cnt_attr) + 1)

    def play_once(self):
        attemp_cnt_1, attemp_cnt_2 = 0, 0

        while attemp_cnt_1 < self.card_id_1:
            self.move(self.grasp_actor(
                self.button1,
                arm_tag="left",
                pre_grasp_dis=0.08,
                grasp_dis=0.08,
                contact_point_id=0,
            ), language_annotation='Press the left button one time.')
            self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation='Press the left button one time.')
            self.update_press_success(self.button1, "press_flag_1", "press_cnt_1")
            self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation='Press the left button one time.')
            self.set_button_unpressed(self.button1)
            self.update_button_reset(self.button1, "press_flag_1")
            attemp_cnt_1 += 1

        while attemp_cnt_2 < self.card_id_2:
            self.move(self.grasp_actor(
                self.button2,
                arm_tag="left",
                pre_grasp_dis=0.08,
                grasp_dis=0.08,
                contact_point_id=0,
            ), language_annotation='Press the middle button one time.')
            self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation='Press the middle button one time.')
            self.update_press_success(self.button2, "press_flag_2", "press_cnt_2")
            self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation='Press the middle button one time.')
            self.set_button_unpressed(self.button2)
            self.update_button_reset(self.button2, "press_flag_2")
            attemp_cnt_2 += 1

        self.move(self.back_to_origin(arm_tag="left"), language_annotation='Press the confirm button.')
        self.move(self.grasp_actor(
            self.check_button,
            arm_tag="right",
            pre_grasp_dis=0.08,
            grasp_dis=0.08,
            contact_point_id=0,
        ), language_annotation='Press the confirm button.')
        self.move(self.move_by_displacement(arm_tag="right", z=-0.045), language_annotation='Press the confirm button.')
        self.update_press_success(self.check_button, "press_flag_check_button", "press_cnt_check_button")

        self.info["info"] = {}
        return self.info

    def check_success(self):
        # update button flag
        self.update_button_reset(self.button1, "press_flag_1")
        self.update_button_reset(self.button2, "press_flag_2")
        self.update_button_reset(self.check_button, "press_flag_check_button")
        # update flag and cnt if pressed successfully
        self.update_press_success(self.button1, "press_flag_1", "press_cnt_1")
        self.update_press_success(self.button2, "press_flag_2", "press_cnt_2")
        self.update_press_success(self.check_button, "press_flag_check_button", "press_cnt_check_button")
        # reset buttons
        self.set_button_unpressed(self.button1, target=min(0.0, self.get_current_button_value("button", self.button1)+0.002))
        self.set_button_unpressed(self.button2, target=min(0.0, self.get_current_button_value("button", self.button2)+0.002))
        self.set_button_unpressed(self.check_button, target=min(0.0, self.get_current_button_value("button", self.check_button)+0.002))

        return self.press_cnt_1 == self.card_id_1 and self.press_cnt_2 == self.card_id_2 and self.press_flag_check_button
        