import numpy as np
import cv2

def angle2R(yz_rotate_angle, xz_rotate_angle, xy_rotate_angle):
    """
    Calculate rotation matrix from three rotation angles (degrees).
    Args:
        yz_rotate_angle (float): Rotation angle around x-axis (degrees)
        xz_rotate_angle (float): Rotation angle around y-axis (degrees)
        xy_rotate_angle (float): Rotation angle around z-axis (degrees)
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    theta2 = np.deg2rad(yz_rotate_angle)
    theta3 = np.deg2rad(xz_rotate_angle)
    theta1 = np.deg2rad(xy_rotate_angle)

    # Rotation matrix around x-axis (YZ plane)
    R2 = np.array([
        [1, 0, 0],
        [0, np.cos(theta2), -np.sin(theta2)],
        [0, np.sin(theta2), np.cos(theta2)]
    ])
    # Rotation matrix around y-axis (XZ plane)
    R3 = np.array([
        [np.cos(theta3), 0, np.sin(theta3)],
        [0, 1, 0],
        [-np.sin(theta3), 0, np.cos(theta3)]
    ])
    # Rotation matrix around z-axis (XY plane)
    R1 = np.array([
        [np.cos(theta1), -np.sin(theta1), 0],
        [np.sin(theta1), np.cos(theta1), 0],
        [0, 0, 1]
    ])
    # Combine the rotation matrices
    rotation_matrix = R1 @ R3 @ R2
    return rotation_matrix

def R2angle(rotation_matrix, case=1):
    """
    Calculate rotation angles from a rotation matrix.
    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix
        case (int): 1 or 2, for two possible solutions
    """
    unit_check = rotation_matrix @ rotation_matrix.T
    print(f"Unit matrix check:\n{unit_check}")

    def calculate_angles(rotation_matrix, y_radians):
        z_radians = np.arcsin(rotation_matrix[1, 0] / np.cos(y_radians))
        x_radians = np.arcsin(rotation_matrix[2, 1] / np.cos(y_radians))
        print(f"\nFor y_radians = {np.rad2deg(y_radians):.10f} degrees:")
        axes = ["x", "y", "z"]
        for axis, angle in zip(axes, [x_radians, y_radians, z_radians]):
            print(f"{axis} :   {np.rad2deg(angle):.10f}   (degree)")

    if case == 1:
        y_radians = np.arcsin(-rotation_matrix[2, 0])
        print(f"y_radians1: {y_radians}")
        calculate_angles(rotation_matrix, y_radians)
    else:
        y_radians = -np.arcsin(rotation_matrix[2, 0])
        print(f"y_radians2: {y_radians}")
        calculate_angles(rotation_matrix, y_radians)

def main():
    print("Select which case to execute:")
    print("1. Calculate rotation matrix from angles (angle2R)")
    print("2. Calculate angles from rotation matrix (R2angle)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("Enter three rotation angles (degrees), separated by spaces:")
        try:
            x, y, z = map(float, input("x y z: ").split())
        except Exception:
            print("Input format error. Please try again.")
            return
        rotation_matrix = angle2R(x, y, z)
        print(f"Rotation matrix:\n{rotation_matrix}")
        np.savetxt("R.csv", rotation_matrix, delimiter=',')
        print("Rotation matrix saved as R.csv")
    elif choice == "2":
        try:
            rotation_matrix = np.loadtxt('needCheck_R/R.csv', delimiter=',')
        except Exception:
            print("Failed to read R.csv. Please run case 1 first or check the file.")
            return
        print("Select calculation method:")
        print("1. case 1 (y_radians = arcsin(-R[2,0]))")
        print("2. case 2 (y_radians = -arcsin(R[2,0]))")
        sub_case = input("Enter 1 or 2: ").strip()
        case_num = 1 if sub_case == "1" else 2
        R2angle(rotation_matrix.T, case=case_num)
    else:
        print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()
