from glob import glob

from setuptools import find_packages, setup

package_name = 'person_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qyang',
    maintainer_email='qiaoyue.yang@uni-bielefeld.de',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'human_tf_node=person_pub.human_tf:main',
            'person_pub=person_pub.person_pub:main',
            'bodyposenet_onnx_node=person_pub.bodyposenet_onnx_node:main',
            'depth_fusion=person_pub.depth_fusion:main',
            'pose_filter_node=person_pub.pose_filter_node:main',
        ],
    },
)
