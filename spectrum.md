# スペクトラグラム

```python
import matplotlib.pyplot as plt
import librosa

wave, rate = librosa.load("sample.wav")
spec = plt.specgram(wave, Fs = rate)
```

![](img/spectrum001.png)