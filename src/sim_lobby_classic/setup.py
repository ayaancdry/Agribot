from setuptools import find_packages, setup
import os 
import glob

package_name = 'sim_lobby_classic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
  #      (os.path.join('share', package_name, 'launch'), ['launch/lobby_classic.launch.py']),
  	(os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), ['worlds/lobby_with_pots.world']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ayaan',
    maintainer_email='ayaan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
