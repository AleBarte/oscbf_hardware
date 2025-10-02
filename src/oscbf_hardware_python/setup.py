from setuptools import find_packages, setup

package_name = "oscbf_hardware_python"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "pyyaml",
    ],
    extras_require={
        "oscbf": ["oscbf @ git+https://github.com/StanfordASL/oscbf"],
    },
    zip_safe=True,
    maintainer="dmorton",
    maintainer_email="danielpmorton@gmail.com",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller = oscbf_hardware_python.scripts.franka_control_node:main",
            "ee_traj_node = oscbf_hardware_python.scripts.traj_node:main",
        ],
    },
)
