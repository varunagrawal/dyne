/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file contact.h
 * @date June 2022
 * @author Varun Agrawal
 * @brief Various utilities related to contact.
 */

#pragma once

namespace dyne {

/// Convenient enum for various contact states. Makes for easy extension.
enum ContactState { SWING = 0, STANCE = 1, SLIPPING = 2 };

/// Convenience enum for the different foot types (point foot, flat foot, etc.)
enum FootType { POINT = 0, FLAT = 1 };

}  // namespace dyne
