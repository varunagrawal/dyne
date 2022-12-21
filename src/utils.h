#include <gtdynamics/universal_robot/Robot.h>
#include <gtdynamics/universal_robot/sdf.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/Values.h>

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace dyne {

/**
 * @brief Utility to read a CSV file. Returns all the data as strings.
 *
 * @param file_path The path to the CSV file.
 * @param delimiter Column delimiter. Default is comma.
 * @return std::vector<std::vector<std::string>>
 */
std::vector<std::vector<std::string>> ReadCsv(
    const std::string& file_path, const std::string& delimiter = ",");

/**
 * @brief Helper class to load Pybullet dataset.
 *
 * Reads a CSV file with headers as:
 * timestamps, wx, wy, wz, ax, ay, az,
 * f1_contact, f2_contact, f3_contact, f4_contact,
 * qw, qx, qy, qz, tx, ty, tz
 *
 */
class DataLoader {
  std::vector<double> timestamps_;
  std::vector<gtsam::Vector3> angular_velocity_;
  std::vector<gtsam::Vector3> linear_acceleration_;
  std::vector<std::map<std::string, int>> contacts_;
  std::vector<gtsam::Values> joint_angles_;

  std::vector<gtsam::NavState> states_;
  std::vector<std::map<std::string, gtsam::Pose3>> feet_poses_;

  std::vector<std::string> feet_;
  gtdynamics::Robot robot_;

  gtsam::Rot3 bRs_;

  using CsvData = std::vector<std::vector<std::string>>;

 public:
  DataLoader() = default;

  DataLoader(const std::string& measurements_file,
             const std::string& ground_truth_file,
             const std::string& robot_file, const std::string& base = "",
             const std::vector<std::string>& feet = {})
      : feet_(feet),
        robot_(gtdynamics::CreateRobotFromFile(robot_file, "", true)) {
    if (!ground_truth_file.empty()) {
      load_ground_truth(ground_truth_file);
    }
    if (!measurements_file.empty()) {
      load_measurements(measurements_file);
    }
  }

  gtsam::Rot3 getRotation(const CsvData& data, uint64_t i,
                          const std::vector<uint64_t>& indices) {
    gtsam::Rot3 R = gtsam::Rot3::Quaternion(
        std::stod(data[i][indices[0]]), std::stod(data[i][indices[1]]),
        std::stod(data[i][indices[2]]), std::stod(data[i][indices[3]]));
    return R;
  }

  gtsam::Point3 getTranslation(const CsvData& data, uint64_t i,
                               const std::vector<uint64_t>& indices) {
    gtsam::Point3 p(std::stod(data[i][indices[0]]),
                    std::stod(data[i][indices[1]]),
                    std::stod(data[i][indices[2]]));
    return p;
  }

  void load_ground_truth(const std::string& ground_truth_file) {
    CsvData data = ReadCsv(ground_truth_file);

    for (size_t i = 1; i < data.size(); i++) {
      gtsam::Rot3 nRb = getRotation(data, i, {1, 2, 3, 4});
      gtsam::Point3 nPb = getTranslation(data, i, {5, 6, 7});
      // Can use the same helper to get velocity as well.
      gtsam::Point3 nVb = getTranslation(data, i, {8, 9, 10});

      states_.emplace_back(nRb, nPb, nVb);

      // Load foot poses
      std::map<std::string, gtsam::Pose3> feet_poses;
      for (size_t f = 0; f < feet_.size(); f++) {
        std::string foot = feet_.at(f);
        feet_poses[foot] = gtsam::Pose3(
            getRotation(data, i,
                        {11 + 7 * f, 12 + 7 * f, 13 + 7 * f, 14 + 7 * f}),
            getTranslation(data, i, {15 + 7 * f, 16 + 7 * f, 17 + 7 * f}));
      }
      feet_poses_.push_back(feet_poses);
    }
  }

  /**
   * @brief Load the measurements data from the measurements file.
   *
   * The measurements are
   * `timestamps,angular_velocity,linear_acceleration,foot_contacts,joint_angles`
   *
   * @param measurements_file
   */
  void load_measurements(const std::string& measurements_file) {
    CsvData data = ReadCsv(measurements_file);

    for (size_t i = 1; i < data.size(); i++) {
      timestamps_.push_back(std::stod(data[i][0]));

      angular_velocity_.push_back(gtsam::Vector3(
          std::stod(data[i][1]), std::stod(data[i][2]), std::stod(data[i][3])));

      linear_acceleration_.push_back(gtsam::Vector3(
          std::stod(data[i][4]), std::stod(data[i][5]), std::stod(data[i][6])));

      std::map<std::string, int> contacts;
      for (size_t fidx = 0; fidx < feet_.size(); fidx++) {
        std::string foot = feet_.at(fidx);
        foot.replace(3, 5, "toe");

        contacts[foot] = std::stoi(data[i][7 + fidx]);
      }
      contacts_.push_back(contacts);

      gtsam::Values joint_angles_at_k;
      for (size_t j = 0; j < robot_.numJoints(); j++) {
        std::string joint_name = data[0][11 + j];
        auto joint = robot_.joint(joint_name);
        uint16_t k = i - 1;
        gtdynamics::InsertJointAngle(&joint_angles_at_k, joint->id(), k,
                                     std::stod(data[i][11 + j]));
      }
      joint_angles_.push_back(joint_angles_at_k);
    }
  }

  size_t size() const { return timestamps_.size(); }

  gtsam::Vector3 angular_velocity(uint64_t index) const {
    return angular_velocity_.at(index);
  }

  gtsam::Vector3 linear_acceleration(uint64_t index) const {
    return linear_acceleration_.at(index);
  }

  gtsam::Values joint_angles(uint64_t index) const {
    return joint_angles_.at(index);
  }

  std::map<std::string, int> contacts(uint64_t index) const {
    return contacts_.at(index);
  }

  gtsam::NavState ground_truth_state(uint64_t index) const {
    return states_.at(index);
  }
};

}  // namespace dyne
