import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
import scipy

fname = 'test_audio\BJL.wav'
outname = 'test_audio\BJL_quanitzation.wav'
# 读取音频文件
sample_rate, audio_data = wavfile.read(fname)
# 将音频数据转换为numpy数组
audio_data = np.array(audio_data, dtype=np.int16)
q_values = [256, 512, 1024]
# 对于每个q值，进行音频量化并保存处理后的音频文件
for q in q_values:
    # 计算量化因子
    quantization_factor = 32768 / q

    # 将音频数据幅度值四舍五入到最接近的q的整数倍
    quantized_audio_data = np.round(audio_data / quantization_factor) * quantization_factor

    # 将处理后的音频数据转换为16位有符号整数
    quantized_audio_data = np.array(quantized_audio_data, dtype=np.int16)

    # 保存处理后的音频文件
    wavfile.write("test_audio\BJL_quanitzation_{}.wav".format(q), sample_rate, quantized_audio_data)