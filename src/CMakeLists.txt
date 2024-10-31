# Copyright 2020, 2021 CNRS - Airbus SAS
# Author: Florent Lamiraux, Joseph Mirabel
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(CMAKE_CXX_STANDARD 14) 
findpython()

install(FILES
  agimus_demos/__init__.py
  agimus_demos/in_state_planner.py
  agimus_demos/tools_hpp.py
  DESTINATION ${PYTHON_SITELIB}/agimus_demos
  )

install(FILES
  agimus_demos/calibration/__init__.py
  agimus_demos/calibration/tooltip_calibration.py
  agimus_demos/calibration/hand_eye_calibration.py
  agimus_demos/calibration/play_path.py
  DESTINATION ${PYTHON_SITELIB}/agimus_demos/calibration
  )

install(FILES
  agimus_demos/talos/tools_hpp.py
  agimus_demos/talos/__init__.py
  DESTINATION ${PYTHON_SITELIB}/agimus_demos/talos
  )

IF(USE_TIAGO)
  find_package(pybind11 CONFIG)
  if(pybind11_FOUND)
    pybind11_add_module(pytsp tsp.cpp tsp-py.cpp)
    target_link_libraries(pytsp PRIVATE Eigen3::Eigen)

    message(STATUS "Installing python module to ${PYTHON_SITELIB}")
    install(TARGETS pytsp
      DESTINATION ${PYTHON_SITELIB}/agimus_demos)
  endif()
ENDIF()

install(FILES
  rqt_agimus/tooltip_calibration.py
  rqt_agimus/contact_calibration.py
  rqt_agimus/camera_calibration.py
  DESTINATION ${PYTHON_SITELIB}/rqt_agimus)

