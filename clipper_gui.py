import sys
from PySide2.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QCheckBox,
    QTextEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide2.QtCore import Qt, QThread, Signal
from clipper import run_clipper


class ClipperWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal()

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        def log(msg):
            self.log_signal.emit(str(msg))

        run_clipper(**self.args, log_callback=log)
        self.done_signal.emit()


class ClipperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETT Video Clipper")
        self.setMinimumWidth(600)
        layout = QVBoxLayout()

        # Video file selection
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        file_btn = QPushButton("Browse...")
        file_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(QLabel("Video File:"))
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)

        # Output name
        outname_layout = QHBoxLayout()
        self.outname_edit = QLineEdit("clips")
        outname_layout.addWidget(QLabel("Output Name:"))
        outname_layout.addWidget(self.outname_edit)
        layout.addLayout(outname_layout)

        # Buffer
        buffer_layout = QHBoxLayout()
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setValue(1.5)
        self.buffer_spin.setSingleStep(0.1)
        buffer_layout.addWidget(QLabel("Time buffer between clips (s):"))
        buffer_layout.addWidget(self.buffer_spin)
        layout.addLayout(buffer_layout)

        # Orderby
        orderby_row = QHBoxLayout()
        self.order_combo = QComboBox()
        self.order_combo.addItems(["Chronological", "Number of shots", "Duration"])
        orderby_row.addWidget(QLabel("Order By:"))
        orderby_row.addWidget(self.order_combo)
        layout.addLayout(orderby_row)

        # Order direction (separate row)
        orderdir_row = QHBoxLayout()
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Ascending", "Descending"])
        orderdir_row.addWidget(QLabel("Order:"))
        orderdir_row.addWidget(self.direction_combo)
        layout.addLayout(orderdir_row)

        # nclips
        nclips_layout = QHBoxLayout()
        self.nclips_spin = QSpinBox()
        self.nclips_spin.setMinimum(0)
        self.nclips_spin.setMaximum(9999)
        self.nclips_spin.setValue(10)
        nclips_layout.addWidget(QLabel("Number of Clips (0=all):"))
        nclips_layout.addWidget(self.nclips_spin)
        layout.addLayout(nclips_layout)

        # starttime
        starttime_layout = QHBoxLayout()
        self.starttime_spin = QSpinBox()
        self.starttime_spin.setMaximum(99999)
        starttime_layout.addWidget(QLabel("Start Time (s):"))
        starttime_layout.addWidget(self.starttime_spin)
        layout.addLayout(starttime_layout)

        # endtime
        endtime_layout = QHBoxLayout()
        self.endtime_spin = QSpinBox()
        self.endtime_spin.setMaximum(99999)
        endtime_layout.addWidget(QLabel("End Time (s):"))
        endtime_layout.addWidget(self.endtime_spin)
        layout.addLayout(endtime_layout)

        # max_time_diff
        maxdiff_layout = QHBoxLayout()
        self.maxdiff_spin = QDoubleSpinBox()
        self.maxdiff_spin.setValue(2.5)
        self.maxdiff_spin.setSingleStep(0.1)
        maxdiff_layout.addWidget(QLabel("Max Time Between Shots (s):"))
        maxdiff_layout.addWidget(self.maxdiff_spin)
        layout.addLayout(maxdiff_layout)

        # delta
        delta_layout = QHBoxLayout()
        self.delta_spin = QDoubleSpinBox()
        self.delta_spin.setValue(0.02)
        self.delta_spin.setSingleStep(0.01)
        delta_layout.addWidget(QLabel("Delta (onset detection):"))
        delta_layout.addWidget(self.delta_spin)
        layout.addLayout(delta_layout)

        # skip_clips
        skip_layout = QHBoxLayout()
        self.skip_edit = QLineEdit()
        skip_layout.addWidget(QLabel("Skip Clips (ids comma separated):"))
        skip_layout.addWidget(self.skip_edit)
        layout.addLayout(skip_layout)

        # Run button
        self.run_btn = QPushButton("Run Clipper")
        self.run_btn.clicked.connect(self.run_clipper)
        layout.addWidget(self.run_btn)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_area)

        self.setLayout(layout)
        self.worker = None

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if fname:
            self.file_edit.setText(fname)

    def run_clipper(self):
        self.log_area.clear()
        ORDER_MAP = {
            "Chronological": "chrono",
            "Number of shots": "shots",
            "Duration": "duration"
        }
        direction = self.direction_combo.currentText()
        descending = direction == "Descending"
        args = {
            "video_file": self.file_edit.text(),
            "buffer": self.buffer_spin.value(),
            "outcsv": False,  # Always False, CSV output removed
            "outclips": self.clips_check.isChecked(),
            "outname": self.outname_edit.text(),
            "orderby": ORDER_MAP[self.order_combo.currentText()],
            "nclips": self.nclips_spin.value() or None,
            "starttime": self.starttime_spin.value(),
            "endtime": self.endtime_spin.value() or None,
            "descending": descending,
            "max_time_diff": self.maxdiff_spin.value(),
            "delta": self.delta_spin.value(),
            "skip_clips": self.parse_skip_clips(self.skip_edit.text()),
        }
        self.run_btn.setEnabled(False)
        self.worker = ClipperWorker(args)
        self.worker.log_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.clipper_done)
        self.worker.start()

    def append_log(self, msg):
        self.log_area.append(msg)

    def clipper_done(self):
        self.run_btn.setEnabled(True)
        self.append_log("\nDone.")

    @staticmethod
    def parse_skip_clips(text):
        if not text.strip():
            return []
        # Accept space or comma separated
        return [int(x) for x in text.replace(",", " ").split() if x.isdigit()]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ClipperGUI()
    gui.show()
    sys.exit(app.exec_())
