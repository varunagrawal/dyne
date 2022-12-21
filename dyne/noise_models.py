"""Various noise models from different robots."""

import numpy as np

from dyne.state_estimator import NoiseParams

a1_noise = NoiseParams(imu_between_sigmas=np.ones(6) * 1e-4,
                       joint_angle=np.deg2rad(1),
                       joint_velocity=3.5e-2,
                       pose=1e-3,
                       contact_point=1e-6,
                       contact_height=1e-6)

anymalc_noise = NoiseParams(bias_prior=np.ones(6) * 1e-5,
                            imu_between_sigmas=np.ones(6) * 1e-3,
                            joint_angle=np.deg2rad(1),
                            joint_velocity=3.5e-2,
                            pose=1e-3,
                            contact_point=1e-6,
                            contact_height=1e-6)

atlas_noise = NoiseParams(bias_prior=np.ones(6) * 1e-5,
                          imu_between_sigmas=np.ones(6) * 1e-3,
                          joint_angle=np.deg2rad(1),
                          joint_velocity=3.5e-2,
                          pose=1e-3,
                          contact_point=1e-6,
                          contact_height=1e-6)
