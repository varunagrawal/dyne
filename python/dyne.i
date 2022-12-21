namespace dyne {

#include <src/noise_params.h>
class NoiseParams {
  NoiseParams(double pose_prior = 0.01, double velocity_prior = 0.01,
              gtsam::Vector6 bias_prior = gtsam::Vector6::Ones() * 1e-3,
              gtsam::Vector6 imu_between_sigmas = gtsam::Vector6::Ones() * 1e-4,
              double joint_angle = 0.0, double joint_velocity = 0.0,
              double pose = 1, double contact_point = 1e-8,
              double contact_height = 1e-8);

  double pose_prior;
  double velocity_prior;
  gtsam::Vector6 bias_prior;
  gtsam::Vector6 imu_between_sigmas;
  double joint_angle;
  double joint_velocity;
  double pose;
  double contact_point;
  double contact_height;
};

#include <src/robot_imu.h>

const double GRAVITY;

class RobotImu {
  RobotImu();

  RobotImu(double freq, double gyroscope_noise_stddev,
           double accelerometer_noise_stddev,
           const gtsam::Pose3& body_P_sensor = gtsam::Pose3(),
           const gtsam::imuBias::ConstantBias& bias =
               gtsam::imuBias::ConstantBias(gtsam::Vector3::Zero(),
                                            gtsam::Vector3::Zero()),
           double gravity = 9.81);

  boost::shared_ptr<gtsam::PreintegrationParams> parse_pim_params(
      double gyroscope_noise_stddev, double accelerometer_noise_stddev,
      const gtsam::Pose3& body_P_sensor = gtsam::Pose3());

  boost::shared_ptr<gtsam::PreintegrationParams> pim_params() const;

  void print() const;

  double freq;
  gtsam::imuBias::ConstantBias previous_bias;
};

#include <src/contact.h>
enum ContactState { SWING, STANCE, SLIPPING };
enum FootType { POINT, FLAT };

#include <src/base.h>
class BaseImu {
  BaseImu();
  BaseImu(const dyne::RobotImu& base_imu, int estimation_rate = 20,
          dyne::NoiseParams noise_params = dyne::NoiseParams());

  gtsam::NonlinearFactorGraph add_prior(uint64_t k,
                                        gtsam::NonlinearFactorGraph& graph,
                                        const gtsam::NavState& state,
                                        bool add_bias_prior = false);
  gtsam::Values insert_values(gtsam::Values& values, uint64_t i,
                              const gtsam::NavState& initial_estimate,
                              bool add_bias = true);

  gtsam::NonlinearFactorGraph add_imu_factor(
      uint64_t i, gtsam::NonlinearFactorGraph& graph,
      const gtsam::PreintegratedImuMeasurements& pim);

  gtsam::NonlinearFactorGraph add_bias_chain(
      uint64_t i, gtsam::NonlinearFactorGraph& graph);

  void set_prior_state(const gtsam::NavState& state_0,
                       bool add_bias_prior = true);

  void step(uint64_t k, const gtsam::Vector3& measured_omega,
            const gtsam::Vector3& measured_acceleration);

  void print() const;

  gtsam::Values update();

  gtsam::Values optimize(bool verbose = false) const;

  gtsam::NonlinearFactorGraph graph();
  gtsam::Values initial();
  void setGraph(const gtsam::NonlinearFactorGraph& graph);
  void setInitial(const gtsam::Values& values);

  dyne::RobotImu base_imu() const;
  gtsam::PreintegratedImuMeasurements pim() const;
  double freq() const;
  double dt() const;
  uint64_t rate() const;
  uint64_t estimation_rate() const;
  gtsam::NavState state() const;
  bool imu_factor_added() const;

  uint64_t index;
};

#include <src/combined_imu_base.h>
class CombinedImuBase {
  CombinedImuBase();
  CombinedImuBase(const dyne::RobotImu& base_imu, int estimation_rate = 20,
                  dyne::NoiseParams noise_params = dyne::NoiseParams());

  gtsam::NonlinearFactorGraph add_prior(uint64_t k,
                                        gtsam::NonlinearFactorGraph& graph,
                                        const gtsam::NavState& state,
                                        bool add_bias_prior = false);
  gtsam::Values insert_values(gtsam::Values& values, uint64_t i,
                              const gtsam::NavState& initial_estimate,
                              bool add_bias = true);

  gtsam::NonlinearFactorGraph add_imu_factor(
      uint64_t i, gtsam::NonlinearFactorGraph& graph,
      const gtsam::PreintegratedCombinedMeasurements& pim);

  void set_prior_state(const gtsam::NavState& state_0,
                       bool add_bias_prior = true);

  void step(uint64_t k, const gtsam::Vector3& measured_omega,
            const gtsam::Vector3& measured_acceleration);

  void print() const;

  gtsam::Values update();

  gtsam::Values optimize(bool verbose = false) const;

  gtsam::NonlinearFactorGraph graph();
  gtsam::Values initial();
  void setGraph(const gtsam::NonlinearFactorGraph& graph);
  void setInitial(const gtsam::Values& values);

  dyne::RobotImu base_imu() const;
  gtsam::PreintegratedCombinedMeasurements pim() const;
  double freq() const;
  double dt() const;
  uint64_t rate() const;
  uint64_t estimation_rate() const;
  gtsam::NavState state() const;
  bool imu_factor_added() const;

  uint64_t index;
};

#include <src/bloesch.h>
class Bloesch : dyne::BaseImu {
  Bloesch();
  Bloesch(const dyne::RobotImu& base_imu, const std::string& model_file,
          uint64_t estimation_rate, const std::string& base_name,
          const std::vector<std::string>& feet,
          const std::map<std::string, gtsam::Point3>& contact_in_com,
          const dyne::NoiseParams& noise_params = dyne::NoiseParams(),
          const dyne::FootType& foot_type = dyne::FootType::POINT,
          bool use_foot_height_constraint = false);
  void add_forward_kinematics_factors(uint64_t k,
                                      const gtsam::Values& joint_angles);
  void add_contact_height_prior(uint64_t k, const std::string& end_link_name,
                                double ground_plane_height = 0.0);
  gtsam::Key add_stance_foot_factor(const std::string& foot);
  void add_factors(uint64_t k, const gtsam::Values& joint_angles,
                   const std::map<std::string, int>& contacts);

  void step(uint64_t k, const gtsam::Vector3& measured_omega_b,
            const gtsam::Vector3& measured_acceleration_b,
            const gtsam::Values& joint_angles,
            const std::map<std::string, int>& contacts);

  std::string base_name() const;
  std::vector<std::string> feet() const;
  gtdynamics::Robot robot() const;
  gtsam::Point3 contact_in_com(const std::string& foot) const;
};

#include <src/dyne.h>
class Dyne : dyne::Bloesch {
  Dyne();
  Dyne(const dyne::RobotImu& base_imu, const std::string& model_file,
       uint64_t estimation_rate, const std::string& base_name,
       const std::vector<std::string>& feet,
       const std::map<std::string, gtsam::Point3>& contact_in_com,
       const dyne::NoiseParams& noise_params = dyne::NoiseParams(),
       const dyne::FootType& foot_type = dyne::FootType::POINT,
       bool use_foot_height_constraint = false);
};

}  // namespace dyne