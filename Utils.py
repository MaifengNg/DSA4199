import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Gradient penalty for WGAN-GP 
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_gen_and_critic(gen_loss_list, critic_loss_list, gan_type, imagePathToSave):
  """
  Save the generator and discriminator loss
  """
  plt.figure(figsize=(10,5))
  
  if gan_type == 'wgan':
    plt.title("Generator and Critic Loss During Training")
    plt.plot(critic_loss_list,label="Critic")
  elif gan_type == 'dcgan':
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(critic_loss_list,label="Discriminator")
    
  plt.plot(gen_loss_list,label="Generator")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(imagePathToSave)
  plt.show()

def save_discriminator_acc(dis_real_acc_list, dis_fake_acc_list, imagePathToSave):
  """
  Save the accuracy of the discriminator on both fake images and real images
  """
  plt.figure(figsize=(10,5))
  plt.title("Discriminator accuracy during training")
  plt.plot(dis_real_acc_list, label="Real covid-19")
  plt.plot(dis_fake_acc_list,label="Fake covid-19")
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig(imagePathToSave)
  plt.show()

def imageNumpyArray(imageTensor):
    """
    Converts a tensor to numpy array
    """
    imageNumpyArray = np.asarray(imageTensor)
    return imageNumpyArray

def saveImage(imageArray, imagePathToSave):
    """
    Save an image array to directory
    """
    fakeImage = imageNumpyArray(imageArray)
    fakeImage = np.transpose(fakeImage, (1,2,0))
    plt.imsave(imagePathToSave, fakeImage, cmap="gray")
    plt.imshow(fakeImage)

def get_dataloader(dataroot, image_size, batch_size, num_workers=2):
    """
    Get dataloader for training GAN model
    """
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    # Create the dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    return loader

def normalizeImage(imageArray):
  """
  Normalise image pixels between -1 and 1
  """
  imageArray = np.asarray(imageArray)
  return (imageArray - np.min(imageArray)) / (np.max(imageArray) - np.min(imageArray))

# Image pixels to histogram
def imageToHistogram(imageToNumpyArray):
  # plot the pixel values
  plt.hist(imageToNumpyArray.ravel(), bins=50, density=True)
  plt.xlabel("pixel values")
  plt.ylabel("relative frequency")
  plt.title("distribution of pixels")