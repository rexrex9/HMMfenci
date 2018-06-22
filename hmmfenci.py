import numpy as np
import pickle
import math

infinite = float(-2 ** 31)
default_train_file_path = './data/msr_training.utf8'
default_model_save_path='./model/HMMfenci.Model'
class FenciModel():
    def __init__(self):
        # 1. 初始化pi、A、B
        self.pi = np.zeros(4)
        self.A = np.zeros((4, 4))
        self.B = np.zeros((4, 65536))
    @staticmethod
    def __log_normalize(a):
        s = 0
        for i in a:
            s += i
        s = math.log(s)
        for i in range(len(a)):
            if a[i] == 0:
                a[i] = infinite
            else:
                a[i] = math.log(a[i]) - s
    def fit(self,train_file_path=default_train_file_path,mode='r', encoding='utf-8'):
        # 1. 加载数据
        with open(train_file_path, mode=mode, encoding=encoding) as reader:
            # 读取所有数据（因为数据格式第一个字符是不可见字符<文件描述符>）
            sentence = reader.read()[1:]
        tokens = sentence.split(' ')
        last_i = 2  # 上一个词结束的状态
        for k, token in enumerate(tokens):
            token = token.strip()
            n = len(token)
            if n <= 0:
                continue
            if n == 1:
                self.pi[3] += 1
                self.A[last_i][3] += 1
                self.B[3][ord(token[0])] += 1
                last_i = 3
                continue
            # 初始化向量
            self.pi[0] += 1  # 作为开始
            self.pi[2] += 1  # 作为结束
            self. pi[1] += (n - 2)  # 中间词数目
            # 转移矩阵
            self.A[last_i][0] += 1
            last_i = 2
            if n == 2:
                self.A[0][2] += 1
            else:
                self.A[0][1] += 1
                self.A[1][1] += (n - 3)
                self.A[1][2] += 1
            # 发射矩阵
            self.B[0][ord(token[0])] += 1
            self.B[2][ord(token[n - 1])] += 1
            for i in range(1, n - 1):
                self.B[1][ord(token[i])] += 1
        # 正则化
        FenciModel.__log_normalize(self.pi)
        for i in range(4):
            FenciModel.__log_normalize(self.A[i])
            FenciModel.__log_normalize(self.B[i])
    def save(self,save_path=default_model_save_path):
        f = open(save_path, 'wb')
        pickle.dump(self, f, 0)
        f.close()
    @staticmethod
    def loadModel(model_path=default_model_save_path):
        with open(default_model_save_path, 'rb') as f:
            fenci = pickle.load(f)
        return fenci
    @staticmethod
    def __go(d, xu, Z, A, B):
        xu=xu[1:]  # 每次迭代消除序列中第一个数字
        if len(xu) == 0:  # 序列中无数字后递归终结
            maxIndex = d.tolist().index(max(d))  # 得到最后最大值所在的序号
            M=[maxIndex]
            for z in reversed(Z):
                maxIndex=z[maxIndex]
                M.append(maxIndex)
            M.reverse()
            return M
        d2 = []  # 用于记录每次deata的取值
        maxIndexs = []  # 用于记录每次最大值的序号
        for i in range(len(A)):  # 循环次数为状态数目
            d1 = d * A[:, i]  # 得到计算中deata的值
            maxvalue = (max(d1))  # 得到计算中deata的最大值
            maxIndex = d1.tolist().index(maxvalue)  # 得到计算中deate最大值所在的序号
            maxIndexs.append(maxIndex)  # 记录计算中deate最大值序号
            d2.append(maxvalue * B[i][ord(xu[0])])  # 得到下一次deata的值
        d2 = np.array(d2)
        Z.append(maxIndexs)  # 记录每次最大值的序号
        return FenciModel.__go(d2, xu, Z, A, B)  # 递归调取自身，传入新的deata代替初始deata
    @staticmethod
    def __getMaxIndex(pi, A, B, xulie):
        d1 = pi * np.transpose(B)[ord(xulie[0])]  # 获取第一次deata的值
        return FenciModel.__go(d1, xulie, [], A, B)
    def __viterbi(self,content):
        return FenciModel.__getMaxIndex(self.pi,self.A,self.B,content)
    @staticmethod
    def __allsegment(sentence, decode): #全模式
        T = len(sentence)
        i = -1
        C=[]
        while i < T-1:  # B/M/E/S
            i += 1
            if decode[i] == 0 or decode[i] == 1:  # Begin
                j = i + 1
                while j < T:
                    if decode[j] == 2:
                        break
                    j += 1
                word=sentence[i:j + 1]
                if word.strip()=='':continue
                C.append(word)
            elif decode[i] == 3 or decode[i] == 2:  # single
                word = sentence[i:i + 1]
                if word.strip() == '': continue
                C.append(word)
            else:
                print("Error")
        return C
    @staticmethod
    def __segment(sentence, decode):#精准模式
        T = len(sentence)
        i = -1
        C = []
        while i < T-1:  # B/M/E/S
            i += 1
            if decode[i] == 0:  # Begin
                j = i + 1
                while j < T:
                    if decode[j] == 2:
                        break
                    j += 1
                word = sentence[i:j + 1]
                if word.strip() == '': continue
                C.append(word)
                i=j
            elif decode[i] == 3 or decode[i] == 2:  # single
                word = sentence[i:i + 1]
                if word.strip() == '': continue
                C.append(word)
            else:
                print("Error")
        return C
    def allfenci(self,content):#全模式
        return FenciModel.__allsegment(content,self.__viterbi(content))
    def fenci(self,content):#精准模式
        return FenciModel.__segment(content,self.__viterbi(content))

if __name__ == '__main__':
    # fenci=FenciModel() #实例化
    # fenci.fit()  #训练模型
    # fenci.save() #保存模型

    content="今天天气非常可以"
    fenci=FenciModel.loadModel()  #加载模型
    print(fenci.fenci(content))

