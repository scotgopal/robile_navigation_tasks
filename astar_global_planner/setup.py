from setuptools import setup
import os
from glob import glob

package_name = 'astar_global_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # plugin description (kept for reference, not used at runtime)
        (os.path.join('share', package_name),
            glob('*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Suhail',
    maintainer_email='suhail@todo.todo',
    description='A* Global Planner for Nav2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar_global_planner = astar_global_planner.astar_global_planner:main',
            'waypoint_follower = astar_global_planner.waypoint_follower:main',
        ],
    },
)
