from PyQt5.QtWidgets import QMainWindow, QApplication, QTextEdit, QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from PyQt5 import QtGui
import sys
from voice import ASR
import warnings
warnings.filterwarnings('ignore')

class Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.run = False
        self.asr = ASR()
        self.initUI()
        self.check = True
    
    def initUI(self):
        self.terminal = QTextEdit(self)
        
        self.terminal.setFixedSize(300, 260)
        self.terminal.setReadOnly(True)
        self.terminal.setText('\nPress Play button to start\n\nAvailable objects:\n-không\n-tôi\n-người\n-một\n')

        self.startBtn = QPushButton()
        self.startBtn.setFixedSize(32,32)
        self.startBtn.setIcon(QtGui.QIcon('button/start.png'))
        self.startBtn.clicked.connect(self.start)
        
        self.stopBtn = QPushButton()
        self.stopBtn.setFixedSize(32,32)
        self.stopBtn.setIcon(QtGui.QIcon('button/stop.png'))
        self.stopBtn.clicked.connect(self.stop)

        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.terminal)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.startBtn)
        self.hbox.addWidget(self.stopBtn)

        self.vbox.addLayout(self.hbox)

        self.wid.setLayout(self.vbox)
        self.setGeometry(100, 100, 300, 300)
        self.setWindowTitle('Speech Project')    
        self.show()

    def start(self):
        self.check = True
        self.loop()
    
    def loop(self):

        if self.check:
            # self.terminal.append('Recording...')
            audios = self.asr.listen()
            # self.terminal.append('Detecting...')
            for audio in audios:
                print(audios.index(audio))
                audio.export('test.wav')
                # self.asr.noise_cancel()
                command = self.asr.predict_word()
            self.loop()
        else:
            print("the end")
        
    def stop(self):
        self.check = False
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    itf = Interface()
    sys.exit(app.exec_())