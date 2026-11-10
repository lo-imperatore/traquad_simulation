from setuptools import find_packages, setup
import os
import re
from glob import glob

package_name = 'rlg_quad_controller'

def model_files():
    """
    Per ogni cartella in models/ installa solo cio' che serve all'inference:
    - rl-games: *.yaml e *.pth
    - run RSL-RL: params/*.yaml e solo l'ultimo checkpoint model_<iter>.pt
    (si escludono gli altri checkpoint, i log tensorboard e git/).
    """
    files = []
    for model_dir in sorted(glob('models/*/')):
        model_dir = model_dir.rstrip('/')
        install_dir = os.path.join('share', package_name, model_dir)
        top = glob(os.path.join(model_dir, '*.yaml')) + glob(os.path.join(model_dir, '*.pth'))
        pts = glob(os.path.join(model_dir, 'model_*.pt'))
        if pts:
            top.append(max(pts, key=lambda p: int(re.search(r'model_(\d+)\.pt$', p).group(1))))
        if top:
            files.append((install_dir, top))
        params = glob(os.path.join(model_dir, 'params', '*.yaml'))
        if params:
            files.append((os.path.join(install_dir, 'params'), params))
    return files

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ] + model_files()

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='tolomei.simone5@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_controller_robot = rlg_quad_controller.inference_controller_robot:main',
            'inference_controller_sim = rlg_quad_controller.inference_controller_sim:main',
            'inference_traquad_sim = rlg_quad_controller.inference_traquad_sim:main',
            'inference_omniquad_robot = rlg_quad_controller.inference_omniquad_robot:main',

            'force_measure = rlg_quad_controller.force_measure:main',
            'trajectory_publisher = rlg_quad_controller.trajectory_publisher:main'
        ],
    },
)
