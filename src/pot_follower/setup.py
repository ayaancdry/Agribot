# ~/sim_ws/src/pot_follower/setup.py

from setuptools import find_packages, setup
import os 
from glob import glob 

package_name = 'pot_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line is good for general files, but not directly needed for console_scripts
    ],
    install_requires=['setuptools', 'rclpy', 'vision_msgs', 'geometry_msgs'], 
    zip_safe=True,
    maintainer='ayaan',
    maintainer_email='ayaan@todo.todo',
    description='ROS 2 package for autonomous flower pot following using YOLO detections.', 
    license='Apache-2.0', # Example license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pot_follower_node = pot_follower.pot_follower_node:main', 
            'extract_images = pot_follower.scripts.extract_images_from_bag:main', 
        ],
    },
)
