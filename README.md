# Deep-Image-Prior-MATLAB-Capstone-Project
This work was submitted under the Math Works Challenge Projects Program. This project implements Deep Image Prior in MATLAB. Image denoising, inpainting and super resolution problems have been solved using DIP in MATLAB.

# Theoretical Framework
Image denoising, inpainting and super resolution are some of the standard problems in digital image processing. Solutions to these problems have usually been obtained in two ways. One is hand crafting an image prior (e.g Total Variation(TV)) i.e manually defining what an image looks like so as to shrink the subset of possible image solutions from the infinite set of images that can be generated. The other is training a neural network on a large dataset of images so as to learn realistic image priors from data. 

Deep Image Prior is a new approach to solving these problems which combines some components of the two outlined methods for obtaining solutions to these problems. With Deep Image Prior the neural network itself acts as an image prior. That is the information contained in the degraded image and the neural network is sufficient to obtain a solution with zero training data, zero validation and test data. There is no need to train the neural network. 
