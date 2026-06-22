import placo
import pickle
from placo_utils.visualization import robot_viz, point_viz
from placo_utils.tf import tf
import numpy as np
import time

from mjlab_microban.robot.microban_constants import HOME_FRAME, MICROBAN_XML

HOME_POSITION = HOME_FRAME.joint_pos

robot = placo.RobotWrapper(str(MICROBAN_XML), placo.Flags.mjcf + placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)

joint_task = solver.add_joints_task()
for dof, angle in HOME_POSITION.items():
    robot.set_joint(dof, angle)
robot.update_kinematics()

trunk_x = 0.01
trunk_y = 0.04
T_world_left_foot = tf.translation_matrix((trunk_x, trunk_y, 0.0))
T_world_right_foot = tf.translation_matrix((trunk_x, -trunk_y, 0.0))


trunk_orientation = solver.add_orientation_task(
    "body", tf.rotation_matrix(0.0, [0, 1, 0])[:3, :3]
)
com = solver.add_position_task("body", np.array([0.0, 0.0, 0.14]))

left_foot_task = solver.add_frame_task("left_foot", T_world_left_foot)
right_foot_task = solver.add_frame_task("right_foot", T_world_right_foot)

jnt = solver.add_joints_task()
jnt.set_joints(
    {
        # "left_Shoulder_Pitch": 0.0000,
        # "left_Shoulder_Roll": -1.3500,
        # "left_Elbow_Pitch": 0.0000,
        # "left_Elbow_Yaw": -0.5000,
        # "right_Shoulder_Pitch": 0.0000,
        # "right_Shoulder_Roll": 1.3500,
        # "right_Elbow_Pitch": 0.0000,
        # "right_Elbow_Yaw": 0.5000,
    }
)

solver.add_regularization_task(1e-6)

for i in range(256):
    solver.solve(True)
    robot.update_kinematics()

robot.set_T_world_frame("left_foot", np.eye(4))
robot.update_kinematics()

t0 = time.monotonic()

viz = robot_viz(robot)
viz.display(robot.state.q)

freq = 50
dt = 1.0 / freq
t = 0
n_steps = 20
duration = n_steps * dt

foot_z = placo.CubicSpline()
foot_z.add_point(0.0, 0.0, 0.0)
foot_z.add_point(duration / 4, 0.03, 0.0)
foot_z.add_point(duration / 2, 0.0, 0.0)

lateral = placo.CubicSpline()
lateral.add_point(0.0, 0.0, 0.0)
lateral.add_point(duration / 4, 0.01, 0.0)
lateral.add_point(duration / 2, 0.0, 0.0)


while True:
    poses = []
    for n_step in range(n_steps):
        t = n_step * dt
        T = t % duration
        if T < duration / 2:
            com_target = com.target_world
            com_target[1] = -lateral.pos(T)
            com.target_world = com_target
            left_foot_task.T_world_frame = tf.translation_matrix(
                (trunk_x, trunk_y, foot_z.pos(T))
            )
            right_foot_task.T_world_frame = tf.translation_matrix(
                (trunk_x, -trunk_y, 0.0)
            )
        else:
            com_target = com.target_world
            com_target[1] = lateral.pos(T - duration / 2)
            com.target_world = com_target
            left_foot_task.T_world_frame = tf.translation_matrix(
                (trunk_x, trunk_y, 0.0)
            )
            right_foot_task.T_world_frame = tf.translation_matrix(
                (
                    trunk_x,
                    -trunk_y,
                    foot_z.pos(T - duration / 2),
                )
            )

        for k in range(10):
            solver.solve(True)
            robot.update_kinematics()

        if n_step == 0:
            print("")
            for dof, angle in HOME_POSITION.items():
                print(f'"{dof}": {robot.get_joint(dof):.4f},')

            T_world_body = robot.get_T_world_frame("body")
            R_body_world = T_world_body[:3, :3].T
            gravity_body = R_body_world[:3, :3] @ np.array([0.0, 0.0, -1.0])
            print("Gravity in the body: ", gravity_body)

        q = [robot.get_joint(dof) for dof in HOME_POSITION.keys()]
        poses.append(q)

        point_viz("CoM", np.array([robot.com_world()[0], robot.com_world()[1], 0.0]))

        T = robot.get_T_world_frame("body")
        print("Body position: ", T[:3, 3])

        viz.display(robot.state.q)
        time.sleep(dt)

    with open("poses.pkl", "wb") as f:
        pickle.dump(poses, f)


point_viz("CoM", np.array([robot.com_world()[0], robot.com_world()[1], 0.0]))

print("Head height: ", robot.get_T_world_frame("head")[2, 3])


T_world_left = robot.get_T_world_frame("left_foot")
T_world_right = robot.get_T_world_frame("right_foot")
T_world_trunk = robot.get_T_world_frame("trunk")
T_trunk_left = np.linalg.inv(T_world_trunk) @ T_world_left
T_trunk_right = np.linalg.inv(T_world_trunk) @ T_world_right

print(T_trunk_left)
print(T_trunk_right)

for dof, angle in HOME_POSITION.items():
    print(f'"{dof}": {robot.get_joint(dof):.4f},')

while True:
    time.sleep(0.1)
