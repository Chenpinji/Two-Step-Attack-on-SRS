import os
import torch
import torchaudio
import scipy.io.wavfile as wf
import numpy as np
import base64
import hashlib
import json 
import time
import uuid
import requests
import scipy
import random
import torch.nn.functional as F
#wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector-step250000.pt").eval()
n_fft = 1024
win_length = None

hop_length = 128
n_mels = 40
sample_rate = 16000

def mel_extraction(wav_tensor1):
    #transform = torchaudio.transforms.MFCC(sample_rate = sample_rate,n_mfcc = 40, melkwargs={"n_fft": 1024, "hop_length": 128, "n_mels": 64, "center": False},)
    transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=False,
    pad_mode="reflect",
    power=1,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,)
    mel_tensor1 = transform(wav_tensor1).T
    # emb_tensor1 = dvector.embed_utterance(mel_tensor1)
    return mel_tensor1
    
def loss_function(AdvTensor, TarTensor):
    # adv_shape = AdvTensor.shape
    # tar_shape = TarTensor.shape
    # print(AdvTensor.size())
    # print(TarTensor.size())
    if len(AdvTensor) > len(TarTensor):
        # print('haha')
        length = AdvTensor.size()[0] - TarTensor.size()[0]
        padding  = [0,0,0,length]
        TarTensor = F.pad(TarTensor, padding, mode='constant', value=0) 
        # print(TarTensor.size())
    elif len(AdvTensor) <= len(TarTensor):
        # print('sef')
        padding  = [0,0,0,TarTensor.size()[0] - AdvTensor.size()[0]]
        AdvTensor = F.pad(AdvTensor, padding, mode='constant', value=0) 
    #     # 如果AdvTensor的维度较长，则裁剪AdvTensor的维度
    #     adv_shape = adv_shape[:len(tar_shape)]
    #     AdvTensor = AdvTensor[tuple(slice(s) for s in adv_shape)]
    # elif len(adv_shape) < len(tar_shape):
    #     # 如果TarTensor的维度较长，则广播AdvTensor的维度
    #     adv_shape = torch.Size([1] * (len(tar_shape) - len(adv_shape))) + adv_shape
    #     AdvTensor = AdvTensor.expand(adv_shape)
    return torch.norm(AdvTensor - TarTensor)
def getSign(timestamp, nonce):
    hs = hashlib.sha256()
    appkey = "zvpcvm5hxib3jz3vt2jshqxjndcywwo2qnx7s6iy"
    secret = '3c8a61e2bdb0ee2d0af74814142ba2ee'
    hs.update((appkey + timestamp + secret + nonce).encode('utf-8'))
    signature = hs.hexdigest().upper()
    return signature
def identifyFeatureByGroupId(confirmFeatureFileName):
    identify_feature = open(confirmFeatureFileName, 'rb').read()
    # 声纹base64字符串
    audio_data = base64.b64encode(identify_feature)
    timestamp = str(int(time.time() * 1000))
    nonce = str(uuid.uuid1()).replace('-', '')
    sign = getSign(timestamp, nonce)
    headers = {"Content-Type": "application/json"}
    appkey = "zvpcvm5hxib3jz3vt2jshqxjndcywwo2qnx7s6iy"
    groupId = '15'
    host = 'https://ai-vpr.hivoice.cn'
    identifyFeatureByGroupIdEndPoint = '/vpr/v1/identifyFeatureByGroupId'
    identify_feature_param = {
        "appkey": appkey,
        "timestamp": timestamp,
        "nonce": nonce,
        "sign": sign,
        "groupId": groupId,
        "topN": 10,
        "audioData": audio_data.decode(),
        "audioSampleRate": 16000,
        "audioFormat": "wav"
    }
    #print('identify_feature_param', identify_feature_param)
    identify_feature_resp = requests.post(url=host + identifyFeatureByGroupIdEndPoint,
                                          data=json.dumps(identify_feature_param),
                                          headers=headers)
    identify_feature_result = json.loads(identify_feature_resp.content)
    return identify_feature_result['data'][0]['featureInfo'], identify_feature_result['data'][0]['score'],identify_feature_result['data'][1]['featureInfo'], identify_feature_result['data'][1]['score']
# for param in wav2mel.parameters():
#     param.requires_grad = False
# for param in dvector.parameters():
#     param.requires_grad = False
path = "/mnt/data/Chenpinji/cmu_dataset"
filename2 = '/mnt/data/Chenpinji/cmu_dataset/cmu7/wav/arctic_a0008.wav'
for cnt in range(0, 4):
    select = random.randint(0,9)
    select2 = random.randint(0,500)
    filename_1 = path + '/' + os.listdir(path)[select] + '/' + 'wav'
    audioname = os.listdir(filename_1)[select2]
    filename1 = filename_1 + '/'+audioname
    wav_tensor1, sample_rate1 = torchaudio.load(filename1, normalize=False)
    wav_tensor2, sample_rate2 = torchaudio.load(filename2, normalize=False)
    noise_tensor = torch.normal(mean = 400., std = 50.,size = (1, len(wav_tensor1[0])))
    wav_tensor1 = wav_tensor1[0].to(torch.float)
    wav_tensor2 = wav_tensor2[0].to(torch.float)
    noise_tensor = noise_tensor[0].to(torch.float)
    #noise_tensor = noise_tensor.cuda()
    noise_tensor.requires_grad = True
    
    
    # wav_tensor1.requires_grad = True
    # wav_tensor2.requires_grad = True
    optimizer = torch.optim.Adam([noise_tensor],lr = 5)
    epochs = random.randint(500,1000)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # AdvTensor = dvector_extraction(wav_tensor1) + dvector_extraction(wav_tensor2)
        # TarTensor = dvector_extraction(noise_tensor)
        #print(TarTensor.grad_fn)
        loss = loss_function(mel_extraction(wav_tensor1 + noise_tensor), mel_extraction(wav_tensor2))
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("epoch={} loss={}".format(epoch, loss))
        if epoch == epochs - 1:
            audio_name = '/mnt/data/Chenpinji/Generate_Adv/mel_Gen1000/'+'melGen_'+str(cnt) + '.wav'
            temp = wav_tensor1 + noise_tensor
            temp = temp.detach().numpy()
            scaled = np.array(np.clip(np.round(temp),-2 ** 15, 2 ** 15 - 1), dtype = np.int16)
            print("epoch={} loss={}".format(epoch, loss))
            wf.write(audio_name, 16000 ,scaled)
    print(cnt)
        
    
