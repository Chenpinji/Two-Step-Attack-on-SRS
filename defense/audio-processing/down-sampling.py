import librosa
import soundfile as sf
import glob  
from tqdm import tqdm 
import os
fname = 'test_audio\BJL.wav'
outname_down = 'test_audio\BJL_downsampling.wav'
outname_recovery = 'test_audio\BJL_recovery.wav'
Target_sr=16000   #目标采样率

audio_origin, sr = librosa.load(fname, sr=None)
print("原始采样率",sr)
print("降采样：目标采样率",Target_sr)
audio_target = librosa.resample(y=audio_origin, orig_sr=sr, target_sr=Target_sr)

#librosa.output.write_wav(outname, audio_target, Target_sr)
sf.write(outname_down, audio_target, Target_sr)

audio_origin,sr1= librosa.load(outname_down, sr=None)
print("升采样（recovery）：目标采样率",sr)#恢复到音频原本的采样率
audio_target = librosa.resample(y=audio_origin, orig_sr=sr1, target_sr=sr)
sf.write(outname_recovery, audio_target, sr)