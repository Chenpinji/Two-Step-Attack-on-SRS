## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import struct
import time
import os
import sys
from collections import namedtuple

sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

import DeepSpeech
#####导入频率掩码
import generate_masking_threshold as generate_mask
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"
timestr = time.time()
namestr = os.path.basename(__file__).split(".")[0]
log_dir = os.path.join('./log', namestr)
log_path = os.path.join(log_dir, str(timestr).split('.')[0] + '.txt')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

audiostr = os.path.join('./adv_audio', namestr)

def convert_mp3(new, lengths):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]),
                               -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i + 2])[0] for i in range(0, len(raw.raw_data), 2)])[
            np.newaxis, :lengths[0]]
    return mp3ed


class Transform(object):
    '''
    Return: PSD
    '''

    def __init__(self, window_size):
        self.scale = 8. / 3.
        self.frame_length = int(window_size)
        self.frame_step = int(window_size // 4)
        self.window_size = window_size

    def __call__(self, x, psd_max_ori):
        win = tf.contrib.signal.stft(x, self.frame_length, self.frame_step)
        z = self.scale * tf.abs(win / self.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10., 9.6) / tf.reshape(psd_max_ori, [-1, 1, 1]) * psd
        return PSD
class Attack:
    def __init__(self, sess, loss_fn, th,psd_max,phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=2000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """

        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3
        self.th=th
        self.psd_max=psd_max


        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.rir = tf.placeholder(tf.float32,name='qq_rir')
        self.pe=tf.placeholder(tf.float32)
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.logits_nora = logits_nora = tf.Variable(np.zeros((113,1, 29), dtype=np.float32), name='qq_logits_nora')
        #113
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_original')
        self.original_co = original_co = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                                     name='qq_co')

        with tf.variable_scope("audio"):
            self.ori_audio= ori_audio=tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_audio')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size, 1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -20000, 20000) * self.rescale
        self.apply_in = original
        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.after_rir = after_rir = self.apply_in + self.create_speech_rir(self.apply_in, self.rir, self.lengths,
                                                                            self.max_audio_len,
                                                                            self.batch_size)
        self.new_input = new_input = self.apply_delta * mask + self.after_rir

        # pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)
        self.pass_in=pass_in = tf.clip_by_value(new_input, -2 ** 15, 2 ** 15 - 1)
        # self.pass_in=pass_in = ori_audio + noise
        # self.pass_in=pass_in = tf.placeholder(tf.float32,[batch_size, max_audio_len])

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(self.pass_in, lengths)
        # And finally restore the graph to make the classifier
        # actually do something interesting.
        batch_size, size = self.ori_audio.get_shape().as_list()
        audio = tf.cast(self.ori_audio, tf.float32)

        # 1. Pre-emphasizer, a high-pass filter
        self.audio = tf.concat(
            (audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1], np.zeros((batch_size, 512), dtype=np.float32)), 1)

        # 2. windowing into frames of 512 samples, overlapping
        self.windowed = tf.stack([audio[:, i:i + 512] for i in range(0, size - 320, 320)], 1)

        window = np.hamming(512)
        self.windoweb = self.windowed * window



        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)

            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input - self.original) ** 2, axis=1) + l2penalty * ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)

        elif loss_fn == "CW":
             ''
        else:
            raise


#########
        self.logits_a=tf.squeeze(self.logits)
        logits_nora_b=tf.squeeze(logits_nora)
        loss_frame=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.logits_a,logits=logits_nora_b))
        loss_frame +=5 * tf.norm(original - original_co,ord=2)+5* tf.norm(original - original_co,ord=1)

        # loss_frame = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_a, labels=logits_nora_c))
        # loss_frame=tf.reduce_mean(tf.square(logits_nora_b-self.logits_a))
        loss_kl= tf.reduce_sum(logits_nora_b*tf.log(logits_nora_b/self.logits_a))
        self.loss_th_list = []
        self.transform = Transform(2048)
        self.psd_max_a = tf.cast(self.psd_max, tf.float32)
        for i in range(self.batch_size):
            logits_delta = self.transform((self.original[i, :] - self.original_co[i, :]), (self.psd_max_a)[i])
            loss_th = tf.reduce_mean(tf.nn.relu(logits_delta - (self.th)[i]))
            loss_th = tf.expand_dims(loss_th, dim=0)
            self.loss_th_list.append(loss_th)
#loss th为心理声学loss
        self.loss_th = tf.concat(self.loss_th_list, axis=0)

        self.loss = loss+self.pe*self.loss_th
        self.loss_frame=loss_frame

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grad, var = optimizer.compute_gradients(self.loss, [original])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])



        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        sess.run(tf.variables_initializer(new_vars + [original]))
        t_vars = tf.trainable_variables()

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

    def create_speech_rir(self,audios, rir, lengths_audios, max_len, batch_size):
        """
        Returns:
            A tensor of speech with reverberations (Convolve the audio with the rir)
        """
        speech_rir = []

        for i in range(batch_size):
            s1 = lengths_audios[i]
            s2 = tf.convert_to_tensor(tf.shape(rir))
            shape = s1 + s2 - 1

            # Compute convolution in fourier space
            sp1 = tf.spectral.rfft(rir, shape)
            sp2 = tf.spectral.rfft(tf.slice(tf.reshape(audios[i], [-1, ]), [0], [lengths_audios[i]]), shape)
            ret = tf.spectral.irfft(sp1 * sp2, shape)

            # normalization
            ret /= tf.reduce_max(tf.abs(ret))
            ret *= 2 ** (16 - 1) - 1
            ret = tf.clip_by_value(ret, -2 ** (16 - 1), 2 ** (16 - 1) - 1)
            ret = tf.pad(ret, tf.constant([[0, 100000]]))
            ret = ret[:max_len]

            speech_rir.append(tf.expand_dims(ret, axis=0))
        speech_rirs = tf.concat(speech_rir, axis=0)
        return speech_rirs

    def attack(self, audio, logit_ori, lengths, target, rir,finetune=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.original_co.assign(np.array(audio)))
        sess.run(self.logits_nora.assign(np.array(logit_ori)))
        sess.run(self.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))

        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t) + [0] * (self.phrase_length - len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size, 1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None] * self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune - audio))

        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iterations
        print('rate:10,no-clip-KL')

        for i in range(MAX):


            index = np.random.randint(0, 512)
            mask_a = np.zeros([1, index])
            mask_b = np.append(mask_a, np.ones([1, 36393 - index]))

            mask_b = np.expand_dims(mask_b, 0)

            sess.run(self.cwmask.assign(mask_b))
            noise = np.random.normal(0, 2000, size=audio.size)
            noise = np.expand_dims(noise, 0)
            sess.run(self.delta.assign(np.array(noise)))

            if i < 500:
                loss, _ = sess.run(
                    [self.loss, self.train], {self.rir: rir[int(i / len(rir))], self.pe: 0.0})
            else:
                dbs = SNR(audio, new_input)
                if dbs < 1:
                    pe = 0.00001
                else:
                    pe = 0.000005
                loss, _ = sess.run(
                    [self.loss, self.train], {self.rir: rir[int(i / len(rir))], self.pe: pe})
            if i%10==0:
                r ,new_input= sess.run([self.decoded,self.original],{self.rir:rir[int(i / len(rir))]})
                dbs=SNR(audio,new_input)
                print("the %d iter Classification:"%i)
                with open(log_path, "a+") as logger:
                    logger.write("the %d iter Classification:\n" % i + "".join([toks[x] for x in r[0].values]))
                    logger.write('\nloss:' + str(loss) + '    shape：' + str(loss.shape)+"DB"+str(dbs))
                print("the %d iter Classification:" % i)
                print("".join([toks[x] for x in r[0].values]))
                print('loss:', loss, 'shape：', loss.shape,'DP:',dbs
                      )
                if i%50==0:
                        audio_name=os.path.join(audiostr,str([toks[x] for x in r[0].values])+str(dbs)+'.wav')
                        wav.write(audio_name, 16000,
                                  np.array(np.clip(np.round(new_input[0]),
                                                   -2 ** 15, 2 ** 15 - 1), dtype=np.int16))



        return final_deltas

def SNR(origanl, current):
    noise=current-origanl

    return 10*np.log10(np.linalg.norm(origanl, ord=2))/np.linalg.norm(noise, ord=2)
def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    #the audio need to play
    lr=100
    l2penalty=float('inf')
    input ='songs/songs/01_original_crafted_fragment_classical_castale.wav'
    # input='adv1.wav'
    normal_audio='init_adv_audio/init_adv_audio/01_call my wife.wav'

    target = 'call my wife '
    out= 'play_music.wav'
    iterations=2000
    restore_path = 'deepspeech-0.4.1-checkpoint/model.v0.4.1'
    outprefix='adv'
    with tf.Session() as sess:
        finetune = []
        audios = []
        # Load the inputs that we're given

        fs, audios = wav.read(input)

        print('source dB', 20 * np.log10(np.max(np.abs(audios))))

        lengths=[]
        lengths.append(len(audios))

        fs,ori_audio= wav.read(normal_audio)
        maxlen_a = max(len(audios),len(ori_audio))
        if len(ori_audio)<len(audios):
            ori_audio=np.append(ori_audio,np.zeros(maxlen_a-len(ori_audio)))
        else:
            audios = np.append(audios, np.zeros(maxlen_a - len(audios)))

        th, psd_max = generate_mask.generate_th(audios.astype(float), fs, 2048)
        audios = np.expand_dims(audios, 0)
        th=np.expand_dims(th,0)
        psd_max=np.expand_dims(psd_max,0)
        ori_audio = np.expand_dims(ori_audio, 0)
        length=max(map(len,audios))
        phrase = target
        logits_ori=np.load('call.npy')
        idx = logits_ori.argmax(axis=2)
        idx=np.squeeze(idx)
        logit=np.eye(29)[idx]
        logit=np.expand_dims(logit, 1)
        root_path = "Room002/"  # 根目录路径
        file_list = []  # 用来存放所有的文件路径
        dir_list = []  # 用来存放所有的目录路径
        get_file_path(root_path, file_list, dir_list)
        # print(file_list)
        irs = []
        Fs=16000
        # for i in range(len(args.impulse)):
        for i in range(len(file_list)):
            # if impulsedir[i].split('.')[-1] == 'wav' or impulsedir[i].split('.')[-1] == 'WAV':
            # fs, ir = wav.read(args.impulse[i])
            if file_list[i].split('.')[-1] == 'wav' or file_list[i].split('.')[-1] == 'WAV':
                fs, ir = wav.read(file_list[i])
                # print('%d audio loaded' % i)
                assert fs == Fs
                irs.append(ir)

        # Pad the impulse responses
        maxlen = max(map(len, irs))
        for i in range(len(irs)):
            irs[i] = np.concatenate((irs[i], np.zeros(maxlen - irs[i].shape[0], dtype=irs[i].dtype)))
        irs = np.array(irs)

        # Set up the attack class and run it
        print('attaked init!')

        # Set up the attack class and run it
        attack = Attack(sess,'CTC',th,psd_max,len(phrase), length,
                        batch_size=len(audios),
                        learning_rate=lr,
                        num_iterations=iterations,
                        l2penalty=l2penalty,
                        restore_path=restore_path)
        deltas = attack.attack(audios,logit,
                               lengths,
                               [[toks.index(x) for x in phrase]],
                               irs,
                               finetune)

        # And now save it to the desired output

        for i in range(len(input)):
            if out is not None:
                path = out
            else:
                path = outprefix + str(i) + ".wav"
            wav.write(path, 16000,
                      np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                       -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
            print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]] - audios[i][:lengths[i]])))


#main()
