from setuptools import setup

package_name = 'dyne'

setup(
    name=package_name,
    version='1.0.0',
    author="Varun Agrawal",
    author_email="varagrawal@gmail.com",
    packages=[package_name],
    data_files=[
        ('data', 'data/' + package_name),
    ],
    zip_safe=False,
    maintainer='Varun Agrawal',
    maintainer_email='varagrawal@gmail.com',
    description='Proprioceptive Legged Robot State Estimator',
    tests_require=[''],
    install_requires=open("requirements.txt").readlines(),
)
