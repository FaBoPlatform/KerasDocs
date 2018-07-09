# Tips

## 色空間の変換と色成分の分解

```python
from matplotlib import pyplot as plt
import cv2

image_path = 'GTSRB/Final_Training/Images/00000/00000_00000.ppm'
image = plt.imread(image_path)
cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
z_image = cvt_image[:,:,2]
plt.imshow(z_image, cmap=plt.cm.gray_r,); 
plt.show()
```

