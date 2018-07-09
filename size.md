# OpenCV

## Sizeの変更

```python
from matplotlib import pyplot as plt
import cv2

image_path = 'GTSRB/Final_Training/Images/00000/00000_00000.ppm'
image = plt.imread(image_path)
cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
resize_image = cv2.resize(cvt_image,(20,20))

plt.imshow(resize_image, cmap=plt.cm.gray_r,); 
plt.show()
```