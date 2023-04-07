from rqt_gui_py.plugin import Plugin
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
import os
from agimus_demos.calibration.play_path import CalibrationControl

class CameraCalibration(Plugin):

    def __init__(self, context):
        super(CameraCalibration, self).__init__(context)
        self.init_calibration()
        
        # Add a group "calibration"
        calib_group = QGroupBox('Calibration')
        calib_layout = QVBoxLayout()
        # Add a button to take a measurement
        calib_layout.addWidget(QPushButton('Take measurement', clicked=self.take_measurement))
        # Add a label that shows the number of measurements taken in a row
        row = QHBoxLayout()
        row.addWidget(QLabel('Measurements taken:'))
        self.measurement_count = QLabel('0')
        row.addWidget(self.measurement_count)
        calib_layout.addLayout(row)

        # Add a row with an input fro selecting a folder and a button to save the measurements
        row = QHBoxLayout()
        self.calib_folder = QLineEdit()
        self.calib_folder.setText(os.getcwd())
        row.addWidget(self.calib_folder)
        row.addWidget(QPushButton('Select folder', clicked=self.select_folder))
        calib_layout.addLayout(row)
        calib_layout.addWidget(QPushButton('Save measurements', clicked=self.save_measurements))

        # add a label to display messages
        self.msg_label = QLabel()
        calib_layout.addWidget(self.msg_label)

        calib_group.setLayout(calib_layout)
        context.add_widget(calib_group)

    def init_calibration(self):
        self.cc = CalibrationControl ()

    def take_measurement(self):
        self.cc.collectData()
        self.measurement_count.setText(str(len(self.cc.measurements)))

    def select_folder(self):
        # ask the user to select a folder
        folder = QFileDialog.getExistingDirectory(None, 'Select folder')
        self.calib_folder.setText(folder)

    def save_measurements(self):
        if self.calib_folder.text() != '':
            self.cc.save(self.calib_folder.text())
            # reset the measurement count
            self.measurement_count.setText('0')
            # Reset the calibration control measurements
            self.cc.measurements = []
            # display a message in green
            self.msg_label.setStyleSheet("QLabel { color : green; }")
            self.msg_label.setText('Measurements saved')
        else:
            # display a message in red
            self.msg_label.setStyleSheet("QLabel { color : red; }")
            self.msg_label.setText('Please select a folder to save the measurements')