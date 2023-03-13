#unisound 打开注释即可使用
import os 
import uuid
import time
import scipy.io.wavfile as wf
import numpy as np
import base64
import hashlib
import json 
import requests
import json
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
    groupId = '015' #15是cmu10  015是vox
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
        "audioFormat": "mp3"
    }
    identify_feature_resp = requests.post(url=host + identifyFeatureByGroupIdEndPoint,
                                          data=json.dumps(identify_feature_param),
                                          headers=headers)
    identify_feature_result = json.loads(identify_feature_resp.content)
    return identify_feature_result['data'][0]['featureInfo'], identify_feature_result['data'][0]['score'],identify_feature_result['data'][1]['featureInfo'], identify_feature_result['data'][1]['score']

def main():
    #输入mp3文件名称，得到该音频在cmu 10人数据集中被分类成啥以及第二可能的人, 在Vox数据集需要改groupid 为015
    file_path='../Voxceleb1_mp3/id10763_00002.mp3'#选择文件
    speaker1,score1,speaker2,score2 = identifyFeatureByGroupId(file_path)
    print('classified as:',speaker1,'Score is:',score1)
    print('second place is:',speaker2, 'Score is:',score2)
main()