# GMM on MFCC

初期評価バージョン。

## GMM on MFCC

それぞれの人物の30秒の音声データを用意する。

```python
from sklearn import preprocessing
import python_speech_features as mfcc
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.mixture import GMM 
import _pickle as pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def wav2mfcc(path):
    audio, rate = librosa.load(path)
    mfcc_data = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy = True)
    mfcc_data = preprocessing.scale(mfcc_data)
    return (mfcc_data, rate)

def drawGraph(data, rate):
    librosa.display.specshow(data, sr=rate, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%02.0f dB')
    plt.tight_layout()

names = ["hozumi", "okuyama", "takeo", "takano"]

for name in names:
    (mfcc_data, rate) = wav2mfcc(name + ".wav")
    gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(mfcc_data)
    pickle.dump(gmm, open(name + ".gmm", 'wb'))

models = []
for name in names:
    model = pickle.load(open(name + ".gmm", 'rb'))
    models.append(model)

(test_data, rate) = wav2mfcc("hozumi_test.wav")
total_scores = np.zeros(len(names)) 
    
for i in range(len(models)):
    gmm    = models[i] 
    scores = np.array(gmm.score(test_data))
    total_scores[i] = scores.sum()
    
speaker = np.argmax(total_scores)
print("Speaker is", names[speaker])
```