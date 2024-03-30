# Session 9 - Assignment

## Basic expectations
- Architecture to C1C2C3C4O (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead)
- Use Dilated kernels here instead of MP or strided convolution for extra points
- Total RF must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- >= 85% accuracy
- As many epochs as you want
- <= 200k params

### Results:
- Epochs: 56
- Parameters: 160,504
- Receptive Field: 247
- Dropout Rate: 0.05
- Training Batch size: 64
- Testing Batch size: 128
- Training
  - Loss=0.5310
  - Accuracy=81.53%
- Testing
  - Average loss: 0.4444
  - Accuracy: 8506/10000 (85.06%)

### To be provided
- Model code
- Torch Summary
- Albumentations code
- Training log
- README.md link
