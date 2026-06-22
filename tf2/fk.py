import numpy as np
import pytorch_kinematics as pk

def fk(joints : np.ndarray, urdf : str = "nico_left_arm.urdf"):
    """
    Calculate position and angle of robot NICOs left hand given a certain joint configuration of NICOs left hand

    Args: 
        th: array of joint positions in degrees. Assumed order: l_shoulder_z, l_shoulder_y, l_arm_x, l_elbow_y, l_wrist_z, l_wrist_x
        urdf: path to urdf file

    Returns:
        Array containing hand_positions x, y, z and euler angles x, y, z in degrees
    """
    chain = pk.build_serial_chain_from_urdf(open(urdf).read(), "left_tcp")
    th = np.deg2rad(joints)
    ret = chain.forward_kinematics(th, end_only=False)
    tg = ret['left_tcp']
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    eul = np.rad2deg(pk.matrix_to_euler_angles(m[:3, :3, :3], "XYZ"))

    return(np.concatenate((pos, eul), axis=None))

# print(fk([-36.5, -38.9, 45.2, 40.2, 29.7, -41.7]))
