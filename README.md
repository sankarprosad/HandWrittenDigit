# Reading a hand-written Bank Slip Without using any OCR or Cognitive Service
https://www.linkedin.com/pulse/reading-hand-written-bank-slip-without-using-any-ocr-chakraborty/


## Published on August 17, 2019
## Author: Sankar Prosad ChakrabortyStatus is online
## Manager Software Development (RPA- Blue Prism & UiPath) at Avanade

### I started with:
... A scanned copy of the actual slip.
... For detecting the digits I compiled a CNN model based on LeNet 5 and trained it with MNIST Hand-Written Digits data.

### Hurdles
... Some digits were even on the Box Boundaries (which is quite normal). But this lead to a high probability of mispredictions as the engine will not able to isolate them, and will assume that part of the digit
... Even after everything, the accuracy of the model was not good.


#### Some digits were even on the Box Boundaries (which is quite normal). But this lead to a high probability of mispredictions as the engine will not able to isolate them, and will assume that part of the digit

**Trick 1**: After trying all possible options, I decided to just take the "Blue" text out of the image, which is possible using OpenCV and which gives an image with just the handwritten text.This is what exactly makes the thing relatively easy. :)

### Even after everything, the accuracy of the model was not good at all :(
This put me under great stress as my whole effort was about to go in vain. The model was detecting 7 as 2, 2 as 7, 9 as 7 blah blah... not at all satisfactory.
So, what actually was wrong in all these?
And here what I found...
**The issue with MNIST:**
Yes, the raw MNIST has some real issues. If you notice the RAW images carefully, you might also notice.
If you carefully look at all the 7s in MNIST, you will see the bottom of most of the 7s are touching the borderline. Whereas for 2 it seldom touches. This kind of bias creates the problem if the data that you are sending is not following the same pattern.

**Trick 2**: As a solution, I did the same preprocessing steps for MNIST data which I used before sending the digits detected from the Bank Slip and recompiled my model with the new dataset.

If you like to embark to a similar kind of project, can refer the following few useful websites/blogs...

https://www.learnopencv.com/ (Highly Recommended)
https://medium.com/@kananvyas
https://medium.com/@surya.kommareddy/number-recognition-using-convolutional-neural-networks-part-1-5dc8a394b0cf
etc.






