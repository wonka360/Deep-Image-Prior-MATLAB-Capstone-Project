# Deep-Image-Prior-MATLAB-Capstone-Project
This work was submitted under the Math Works Challenge Projects Program. This project implements Deep Image Prior in MATLAB. Image denoising, inpainting and super resolution problems have been solved using DIP in MATLAB.

# Theoretical Framework
Image denoising, inpainting and super resolution are some of the standard problems in digital image processing. Solutions to these problems have usually been obtained in two ways. One is hand crafting an image prior (e.g Total Variation (TV)) i.e manually defining what an image looks like so as to shrink the subset of possible image solutions from the infinite set of images that can be generated. The other is training a neural network on a large dataset of images so as to learn realistic image priors from data. 

Deep Image Prior is a new approach to solving these problems which combines some components of the two outlined methods for obtaining solutions to these problems. With Deep Image Prior the neural network itself acts as an image prior. That is the information contained in the degraded image and the neural network is sufficient to obtain a solution with zero training data, zero validation and test data. There is no need to train the neural network. 

Suppose we have the image degradation model: $y = A(x) + n$ where $y$ is the degraded image.

Where $x$ is a clean image that was passed through the degradation function $A()$. There are infinitely many clean images that can take on the value of $x$ to produce $y$. The objective is to recover the clean image $x$ given the degraded image $y$

This can be done by formulating an objective optimization function which consists of a data fidelity term and a Regularization term

```math
\hat{x} = \displaystyle\arg\min_x \underbrace{\|A(x) - y\|^2}_{\text{Data Fidelity}} + \lambda \underbrace{R(x)}_{\text{Regularization Term}}
```

The data fidelity term measures how closely the estimated image $x$ is to the degraded image $y$. This term is to be minimized.

The regularization term is usually a mathematical model which rules out uncharacteristic images. 

In Deep Image prior the CNN itself acts as an image prior, this is because CNNs offer a high impedance to noise. Low frequency structural components are fitted first then later high frequency components after many iterations. The CNN can easily be regularized through early stopping that is stopping the optimization process as high frequency components are about to be fitted. 

The optimization problem for DIP is then formulated as such

```math
R(x) = 
\begin{cases} 
0, & \text{if } x = f_\theta(z) \\ 
\infty, & \text{otherwise} 
\end{cases}
```

where $f_\theta(\cdot)$ is the CNN parameterized by $\theta$, and $z$ is a random noise tensor given as input to the network. $x_o$ is the degraded image

```math
\theta^* = \arg\min\limits_\theta \|A(f_\theta(z)) - x_o\|^2
```
We insert the random noise tensor $z$ into the CNN and search for a solution in the space spanned by the neural network architecture by minimizing $\theta$

**Denoising**

For Denoising no spatial degradation occurs so $A$ = $I$ where $I$ is the identity operator. So $x_o$ = $x$ + $n$ where $n$ is any random noise.

The problem is formulated as:

```math
\theta^* = \arg\min\limits_\theta \|f_\theta(z) - x_o\|^2
```
Noisy Image

<img width="256" height="256" alt="fruitsNoisy" src="https://github.com/user-attachments/assets/d9050f40-a119-445f-b124-68f71686b12b" />

Denoised Image through DIP

<img width="256" height="256" alt="output_image_denoised (1)" src="https://github.com/user-attachments/assets/04060dc5-427b-42de-9403-37092319e138" />

**Inpainting**

For inpainting $A$ = $M$ (a binary mask operator). $A$ zeros out corrupted pixel locations

The problem is formulated as

```math
\theta^* = \arg\min\limits_\theta \|M \odot f_\theta(z) - x_o\|^2
```
In painted Image

<img width="256" height="256" alt="inpainted_image" src="https://github.com/user-attachments/assets/785a9812-e944-477a-b3f1-bdb84425f58b" />

Results after passing through DIP

<img width="256" height="256" alt="output_image_depainted" src="https://github.com/user-attachments/assets/3d2134ba-0db2-4a5b-9dcf-0ebbfb78d28a" />

**Super Resolution**

For super resolution $A$ = $d$ a down sampling operator where $d: \mathbb{R}^{rH \times rW \times C} \to \mathbb{R}^{H \times W \times C}$

```math
\theta^* = \arg\min\limits_\theta \|d(f_\theta(z)) - x_o\|^2
```

Low Resolution Image

<img width="1500" height="1000" alt="child" src="https://github.com/user-attachments/assets/91689249-2d0e-4845-aefc-f0e1d4c2b775" />

High Resolution

<img width="512" height="512" alt="output_image_superres" src="https://github.com/user-attachments/assets/814d1937-fe62-4485-85ae-ee2b63246382" />



                                            


                   



