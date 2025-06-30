# ~/sim_ws/src/pot_follower/setup.py

from setuptools import find_packages, setup
import os # Import os for data_files if you add them later
from glob import glob # Import glob for data_files if you add them later

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
        # If you added the 'scripts' folder, you might have this:
        # (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools', 'rclpy', 'vision_msgs', 'geometry_msgs'], # Ensure these are here
    zip_safe=True,
    maintainer='ayaan',
    maintainer_email='ayaan@todo.todo',
    description='ROS 2 package for autonomous flower pot following using YOLO detections.', # More descriptive
    license='Apache-2.0', # Example license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pot_follower_node = pot_follower.pot_follower_node:main', # <--- ADD THIS LINE
            'extract_images = pot_follower.scripts.extract_images_from_bag:main', # <--- ADD THIS LINE (optional, but useful)
        ],
    },
)
