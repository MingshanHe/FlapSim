import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 用户参数
# ============================================================

BASE_POSITION = np.array([0.0, 0.0, 0.5])

DURATION = 20.0  # 测试持续时间，单位：s
FREQUENCY_HZ = 0.25  # 正弦运动频率，单位：Hz

DEFAULT_HINGE_AMPLITUDE = np.deg2rad(20.0)  # 无关节限制时，默认转动幅值
DEFAULT_SLIDE_AMPLITUDE = 0.02  # 无关节限制时，默认移动幅值，单位：m

USE_PHASE_OFFSET = True  # True：每个关节具有不同相位
REAL_TIME = True  # 是否按照真实时间播放


def get_object_name(model, obj_type, obj_id, default_name):
    """安全获取 MuJoCo 对象名称。"""
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else default_name


def find_robot_root_body(model):
    """
    找到机器人最上层的 body。

    从有关节的 body 开始，沿 parent body 向上查找，
    直到找到直接连接 worldbody 的 body。
    """
    if model.njnt == 0:
        return None

    root_candidates = set()

    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])

        while model.body_parentid[body_id] != 0:
            body_id = int(model.body_parentid[body_id])

        root_candidates.add(body_id)

    if not root_candidates:
        return None

    # 对普通单机器人 URDF，通常只有一个根 body
    return sorted(root_candidates)[0]


def configure_base(model, data, target_position):
    """
    将机器人基座固定在指定位置。

    情况1：机器人有 free joint
        每个循环都锁定 free joint 的 qpos 和 qvel。

    情况2：机器人是固定基座
        直接修改根 body 相对于 worldbody 的 body_pos。
    """
    free_joint_id = None

    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = joint_id
            break

    if free_joint_id is not None:
        qpos_address = int(model.jnt_qposadr[free_joint_id])
        dof_address = int(model.jnt_dofadr[free_joint_id])

        # free joint 的 qpos:
        # [x, y, z, qw, qx, qy, qz]
        root_pose = data.qpos[qpos_address : qpos_address + 7].copy()
        root_pose[0:3] = target_position

        # 防止 URDF 初始四元数异常
        quaternion_norm = np.linalg.norm(root_pose[3:7])

        if quaternion_norm < 1e-8:
            root_pose[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            root_pose[3:7] /= quaternion_norm

        free_joint_name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            free_joint_id,
            f"free_joint_{free_joint_id}",
        )

        print(
            f"[Base] 检测到 free joint: {free_joint_name}\n"
            f"[Base] 将其固定在: {target_position}"
        )

        return {
            "type": "free_joint",
            "joint_id": free_joint_id,
            "qpos_address": qpos_address,
            "dof_address": dof_address,
            "root_pose": root_pose,
        }

    # 没有 free joint，说明通常是固定基座模型
    root_body_id = find_robot_root_body(model)

    if root_body_id is None:
        raise RuntimeError("没有找到机器人的根 body。")

    root_body_name = get_object_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_id,
        f"body_{root_body_id}",
    )

    model.body_pos[root_body_id, :] = target_position

    print(
        f"[Base] 模型为固定基座模型。\n"
        f"[Base] 根 body: {root_body_name}\n"
        f"[Base] 根 body 位置设置为: {target_position}"
    )

    return {
        "type": "fixed_body",
        "body_id": root_body_id,
    }


def lock_base(data, base_information):
    """在每一个控制周期中锁定 free joint 基座。"""
    if base_information["type"] != "free_joint":
        return

    qpos_address = base_information["qpos_address"]
    dof_address = base_information["dof_address"]
    root_pose = base_information["root_pose"]

    data.qpos[qpos_address : qpos_address + 7] = root_pose

    # free joint 有6个速度自由度：
    # vx, vy, vz, wx, wy, wz
    data.qvel[dof_address : dof_address + 6] = 0.0


def collect_scalar_joints(model, data):
    """
    收集可以使用标量正弦轨迹驱动的关节。

    支持：
        hinge：revolute / continuous
        slide：prismatic

    暂时跳过：
        free joint
        ball joint
    """
    controlled_joints = []

    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]

        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue

        joint_name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
            f"joint_{joint_id}",
        )

        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])

        initial_position = float(data.qpos[qpos_address])
        is_limited = bool(model.jnt_limited[joint_id])

        if is_limited:
            lower_limit = float(model.jnt_range[joint_id, 0])
            upper_limit = float(model.jnt_range[joint_id, 1])

            center = 0.5 * (lower_limit + upper_limit)

            # 取整个运动范围的40%，这样不会碰到极限位置
            amplitude = 0.4 * (upper_limit - lower_limit)
        else:
            lower_limit = None
            upper_limit = None
            center = initial_position

            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                amplitude = DEFAULT_HINGE_AMPLITUDE
            else:
                amplitude = DEFAULT_SLIDE_AMPLITUDE

        controlled_joints.append(
            {
                "id": joint_id,
                "name": joint_name,
                "type": joint_type,
                "qpos_address": qpos_address,
                "dof_address": dof_address,
                "center": center,
                "amplitude": amplitude,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
            }
        )

    return controlled_joints


def print_joint_information(controlled_joints):
    print("\n========== Controlled joints ==========")

    for index, joint in enumerate(controlled_joints):
        if joint["type"] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_type_name = "hinge"
            center_text = f"{np.rad2deg(joint['center']):.2f} deg"
            amplitude_text = f"{np.rad2deg(joint['amplitude']):.2f} deg"
        else:
            joint_type_name = "slide"
            center_text = f"{joint['center']:.4f} m"
            amplitude_text = f"{joint['amplitude']:.4f} m"

        print(
            f"{index:02d} | "
            f"{joint['name']:<25} | "
            f"{joint_type_name:<6} | "
            f"center={center_text:<12} | "
            f"amplitude={amplitude_text}"
        )

    print("=======================================\n")


def run_sine_test(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    base_information = configure_base(
        model=model,
        data=data,
        target_position=BASE_POSITION,
    )

    controlled_joints = collect_scalar_joints(model, data)

    if not controlled_joints:
        raise RuntimeError("模型中没有找到 hinge 或 slide 类型的可控关节。")

    print_joint_information(controlled_joints)

    number_of_joints = len(controlled_joints)

    if USE_PHASE_OFFSET:
        phases = np.linspace(
            0.0,
            2.0 * np.pi,
            number_of_joints,
            endpoint=False,
        )
    else:
        phases = np.zeros(number_of_joints)

    angular_frequency = 2.0 * np.pi * FREQUENCY_HZ

    time_history = []
    position_history = []

    start_time = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start_time = time.perf_counter()
            elapsed_time = loop_start_time - start_time

            if elapsed_time >= DURATION:
                break

            # ------------------------------------------------
            # 1. 固定机器人基座
            # ------------------------------------------------
            lock_base(data, base_information)

            current_positions = []

            # ------------------------------------------------
            # 2. 给每个标量关节设置正弦轨迹
            # ------------------------------------------------
            for joint_index, joint in enumerate(controlled_joints):
                phase = phases[joint_index]

                desired_position = joint["center"] + joint["amplitude"] * np.sin(
                    angular_frequency * elapsed_time + phase
                )

                desired_velocity = (
                    joint["amplitude"]
                    * angular_frequency
                    * np.cos(angular_frequency * elapsed_time + phase)
                )

                qpos_address = joint["qpos_address"]
                dof_address = joint["dof_address"]

                data.qpos[qpos_address] = desired_position
                data.qvel[dof_address] = desired_velocity

                current_positions.append(desired_position)

            # 直接修改 qpos 后，需要调用 mj_forward
            # 重新计算 body、geom、site 等的位置
            mujoco.mj_forward(model, data)

            # ------------------------------------------------
            # 3. 记录关节轨迹
            # ------------------------------------------------
            time_history.append(elapsed_time)
            position_history.append(current_positions)

            # ------------------------------------------------
            # 4. 更新 MuJoCo Viewer
            # ------------------------------------------------
            viewer.sync()

            if REAL_TIME:
                calculation_time = time.perf_counter() - loop_start_time
                remaining_time = model.opt.timestep - calculation_time

                if remaining_time > 0.0:
                    time.sleep(remaining_time)

    return (
        np.asarray(time_history),
        np.asarray(position_history),
        controlled_joints,
    )


def plot_joint_trajectories(
    time_history,
    position_history,
    controlled_joints,
):
    """分别绘制转动关节和移动关节的轨迹。"""
    hinge_indices = []
    slide_indices = []

    for index, joint in enumerate(controlled_joints):
        if joint["type"] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_indices.append(index)
        elif joint["type"] == mujoco.mjtJoint.mjJNT_SLIDE:
            slide_indices.append(index)

    if hinge_indices:
        plt.figure(figsize=(11, 6))

        for index in hinge_indices:
            plt.plot(
                time_history,
                np.rad2deg(position_history[:, index]),
                label=controlled_joints[index]["name"],
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Joint angle (deg)")
        plt.title("Hinge-joint sinusoidal trajectories")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    if slide_indices:
        plt.figure(figsize=(11, 6))

        for index in slide_indices:
            plt.plot(
                time_history,
                position_history[:, index],
                label=controlled_joints[index]["name"],
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Joint displacement (m)")
        plt.title("Slide-joint sinusoidal trajectories")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Drive MuJoCo URDF joints using sinusoidal trajectories."
    )

    parser.add_argument(
        "model",
        type=Path,
        help="URDF 文件路径，例如 robot.urdf",
    )

    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")

    time_history, position_history, controlled_joints = run_sine_test(args.model)

    plot_joint_trajectories(
        time_history,
        position_history,
        controlled_joints,
    )


if __name__ == "__main__":
    main()
