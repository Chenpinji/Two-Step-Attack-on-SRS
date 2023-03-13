import hashlib
import requests
URL = 'http://119.3.22.24:3997/'
url_lst = ['detectregister', 'registeruser', 'addsample', 'detectquery','trainmodel', 'deleteuser', 'verifymodel', 'identifymodel']
def get_str_md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()
class TalentedSoft:
    def __init__(self,content,token,userid):
        self.body = {
            'userid': userid,
            'token': token,
            'content': content,
            'step': '0',
        }
        self.files = {
            'file': '0',
        }

    def url_req(self, userid, req, uploadfile):
        self.body['userid'] = userid
        if req == 2:
            for i in range(5):
                self.body['step'] = f'{i + 1}'
                self.files['file'] = open(uploadfile[i], 'rb')
                r = requests.post(URL + url_lst[req], data=self.body, files=self.files)
                print(r.text)
        elif (req == 6) | (req == 7):
            self.files['file'] = open(uploadfile, 'rb')
            r = requests.post(URL + url_lst[req], data=self.body, files=self.files)
            return r.text
        else:
            r = requests.post(URL + url_lst[req], data=self.body)
            print(r.text)

def talent_score(testfile,userid):#这个函数是我自己加的，可以用来获取得分,是进行1:n比对用的，若要进行1：1比对需要把req改成6，同时userid要填写被比对的人
    req = 7
    content = 'arctic_a0008.wav'
    token = get_str_md5('Chenpinji&CMU') #Chenpinji&Cpjkey是Vox数据集(应该是word里面那组)  Chenpinji&CMU是cmu数据集
    ts = TalentedSoft(content,token,userid)
    ans = ts.url_req(userid, req, testfile)
    print(ans)
    return ans.split(",")[1].split("[")[1].split(" ")[0], float(ans.split(",")[4].split(":")[1][1:6])

def talent_sv(testfile, userid):
    req = 6
    content = 'arctic_a0008.wav'
    token = get_str_md5('Chenpinji&CMU') #Chenpinji&Cpjkey是Vox数据集(应该是word里面那组)  Chenpinji&CMU是cmu数据集  
    ts = TalentedSoft(content,token,userid)
    ans = ts.url_req(userid, req, testfile)
    print(ans)
    return float(ans.split(":")[5][1:6])

audio_name = "../cmu_dataset/cmu2/wav/arctic_a0008.wav"
asperson, s = talent_score(audio_name,'cmu7') #这个第二个参数userid在1：n比对没有用处
print('classified as:',asperson,'Score is:',s)
#print('second place is:',speaker2, 'Score is:',score2)