import sounddevice as sd
import soundfile as sf
import time
import queue
import numpy

q = queue.Queue()



def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def recording_file(filename, sentence, fileText, name):
    try:
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(filename, mode='x', samplerate=22000,
                        channels=1) as file:
            with sd.InputStream(samplerate=22000, device=sd.default.device,
                                channels=1, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))

    file = open(fileText, "a")
    file.write(name + '\n')
    file.write(sentence + '\n')
    file.close()

name='cau_10.wav'
fileText='./file/tam_su/tam_su.txt'
sentence='Anh  cần 3 tháng để thích nghi, nếu không sẽ dọn ra ngoài ở'
filename='./file/tam_su/' + name

# HLV đội khách Nguyễn Thanh Sơn ngay lập tức chỉ 
# đạo hậu vệ phải Hồ Tấn Tài hạn chế dâng cao tham gia tấn công, tập trung cho nhiệm vụ theo kèm cầu thủ từng giành HC bạc U23 châu Á năm 2018
recording_file(filename, sentence, fileText, name)