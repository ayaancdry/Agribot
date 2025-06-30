import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory # This is not strictly needed if using FindPackageShare with Path()
from pathlib import Path # <--- MAKE SURE TO ADD THIS IMPORT

def generate_launch_description():
    # Path to your custom world
    world_path = os.path.join(
        FindPackageShare('sim_lobby_classic').find('sim_lobby_classic'),
        'worlds',
        'lobby_with_pots.world'
    )
    
    # Correct path to sim_bot's four-wheel XACRO
    xacro_file = os.path.join(
        FindPackageShare('sim_bot').find('sim_bot'),
        'description', # matches your path "sim_ws/src/sim_bot/description"
        'four_wheel.urdf.xacro'
    )
    robot_desc = Command(['xacro ', xacro_file])

    # Define path to your TRAINED YOLO model (CRITICAL: ENSURE THIS PATH IS CORRECT!)
    # This points to where you copied your 'best.pt' renamed to 'flower_pot_detector.pt'
    yolo_model_path = os.path.expanduser('~/sim_ws/custom_yolo_models/flower_pot_detector.pt') # <--- ADD THIS LINE


    return LaunchDescription([
        # 1️⃣ Launch Gazebo Classic with ROS plugin  
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose', world_path,
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so'
            ],
            output='screen'
        ),

        # 2️⃣ Publish robot description from XACRO
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),

        # 3️⃣ Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'four_wheel_bot',
                '-topic', 'robot_description',
                '-x', '0', '-y', '0', '-z', '0.1'
            ],
            output='screen'
        ),


        # 4️⃣ YOLO Detection Node (ADD THIS BLOCK)
        # Include the yolo.launch.py from yolo_bringup, passing arguments
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                str(Path(FindPackageShare('yolo_bringup').find('yolo_bringup'))),
                '/launch/yolo.launch.py'
            ]),
            launch_arguments={
                'model': yolo_model_path,
                'input_topic': '/front_camera/image_raw',
                'publish_debug_image': 'True',
                'device': 'cpu',
                'conf': '0.25', # Keep this from previous debugging
                'use_yolo_msg_type': 'False', # <--- ADD THIS CRUCIAL LINE
                'namespace': '', # <--- ADD OR CHANGE THIS LINE TO AN EMPTY STRING
            }.items()
        ),

        # 5️⃣ Autonomous Pot Follower Node (ADD THIS BLOCK)
        Node(
            package='pot_follower',
            executable='pot_follower_node',
            output='screen',
            parameters=[{
                'target_pot_class': 'flower_pot',
                'linear_speed': 0.2,
                'angular_speed_gain': 0.5,
                'center_tolerance_x': 0.1,
                'max_pot_height_for_stop': 0.7,
                'image_width': 640.0,
                'image_height': 480.0,
                'detection_timeout_sec': 0.5
            }]
        ),
    ])
