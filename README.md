# Federated Adversarial Defense via Adversarial Training and Pruning Against Backdoor Attack

This project demonstrates the use of adversarial defense mechanisms in deep learning models, particularly adversarial training and pruning strategies to defend against backdoor attacks in a federated learning setup. The code specifically utilizes adversarial attacks and defense methods evaluate the effectiveness of the model under attack conditions. It supports multiple datasets: CIFAR-10, CIFAR-100, and MNIST.

Features Simple CNN Model A simple Convolutional Neural Network (CNN) model for image classification on the CIFAR-10, CIFAR-100, or MNIST datasets. Data Augmentation: Implemented various data augmentation techniques, including random rotations, flips, and brightness adjustments to improve model generalization. Adversarial Attack: Utilizes FGSM (Fast Gradient Sign Method) to generate adversarial examples and evaluate the model's robustness. Flame Flair Comparison: Uses the Structural Similarity Index (SSIM) to compare original and adversarial images, providing a flame flair comparison to assess the effects of adversarial attacks.

Datasets The following datasets are supported in this project: CIFAR-10 A dataset with 60,000 32x32 color images in 10 different classes (training set: 50,000 images; test set: 10,000 images). CIFAR-100: A dataset similar to CIFAR-10 but with 100 classes. MNIST: A dataset with 60,000 28x28 grayscale images of handwritten digits (0-9). Requirements The following Python packages are required to run the code: torch (PyTorch) torchvision art (Adversarial Robustness Toolbox) numpy matplotlib scikit-image PIL (Pillow)

You can install these dependencies using pip:

pip install torch torchvision art numpy matplotlib scikit-image Pillow
pip install torch torchvision art numpy matplotlib scikit-image Pillow

### Key Updates:
1. **Multiple Datasets**: The `README` now reflects support for CIFAR-10, CIFAR-100, and MNIST datasets.
2. **Dataset Loading**: Instructions on how to select the dataset in the code (`'CIFAR10'`, `'CIFAR100'`, or `'MNIST'`).
3. **How to Run**: Provides clear steps to run the code using different datasets.

