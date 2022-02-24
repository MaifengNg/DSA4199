from Utils import saveImage, imageNumpyArray, save_gen_and_critic, get_dataloader, save_discriminator_acc
from ModelDcgan import Generator, Discriminator, weights_init
import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import random

def train_dcgan(image_size, batch_size, epochs, device, loader, path_save, resume_training, save_model, every_100_epochs, num_images_generate):
    """
    Initialise the seeds to get consistent results
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    if (device == 'gpu' and not torch.cuda.is_available()):
        print(f'device given {device} but cuda not available. Using cpu to train model.')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = 1 if device == 'cuda' else 0

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    LEARNING_RATE = 1e-4

    BATCH_SIZE = batch_size
    IMAGE_SIZE = image_size
    NUM_EPOCHS = epochs

    CHANNELS_IMG = 3
    Z_DIM = 100
    NUMBER_OF_IMAGES_TO_GENERATE = num_images_generate

    '''
    To initialise the feature extractor for the generator and the discriminator
    based on the size of the image to generate
    '''
    if image_size == 128:
        ngf = 128
        ndf = 32
    elif image_size == 64:
        ngf = 64
        ndf = 64
    
    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Create the generator
    netG = Generator(ngpu, Z_DIM, CHANNELS_IMG, ngf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu, Z_DIM, CHANNELS_IMG, ndf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if resume_training:
        modelPath = os.path.join(path_save, 'model.tar')
        checkpoint = torch.load(modelPath)
        netG.load_state_dict(checkpoint['generator'])
        netD.load_state_dict(checkpoint['discriminator'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

        netG.train()
        netD.train()

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_real_acc = []
    D_fake_acc = []

    print()
    print('Generator summary')
    summary(netG, (Z_DIM, 1, 1))
    print('Discriminator summary')
    summary(netD, (CHANNELS_IMG, image_size, image_size))
    print()

    """
    Create dir to save gen-dis loss and fake images generated every 10 epochs
    """
    loss_plot_dir = os.path.join(path_save, f'loss_plot')
    try:
        os.mkdir(loss_plot_dir) 
    except FileExistsError:
        print(f'{loss_plot_dir} exist. Overwriting exisiting files.')

    fake_image_dir = os.path.join(path_save, f'fake_images_per_10_epochs')
    try:
        os.mkdir(fake_image_dir) 
    except FileExistsError:
        print(f'{fake_image_dir} exist. Overwriting exisiting files.')

    dis_acc_plot_dir = os.path.join(path_save, f'dis_acc_plot')
    try:
        os.mkdir(dis_acc_plot_dir) 
    except FileExistsError:
        print(f'{dis_acc_plot_dir} exist. Overwriting exisiting files.')

    new_dir = os.path.join(path_save, f'generated_images')
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        print(f'{new_dir} exist. Overwriting exisiting files.')

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(1, NUM_EPOCHS+1):
        print(f'epoch {epoch}')
        # For each batch in the dataloader
        for batch_idx, (real, _) in enumerate(loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = real.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Add discriminator accuracy on real covid-19 images to list
            D_real_acc.append(D_x)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # add discriminator accuracy on fake images to list
            D_fake_acc.append(D_G_z2)

            # Output training stats
            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, batch_idx, len(loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        """
        Saving model at current epoch 
        """
        if save_model:
            modelPathSave = os.path.join(path_save, 'model.tar')
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                }, modelPathSave)

        if epoch >= 0:
            if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                # Save gen and discri loss plot
                gen_and_discri_image_path = os.path.join(loss_plot_dir, f'epoch_{epoch}_gen_discri_loss.png')
                save_gen_and_critic(G_losses, D_losses, 'dcgan', gen_and_discri_image_path)

                # save accuracy scores
                dis_real_image_path = os.path.join(dis_acc_plot_dir, f'epoch_{epoch}_dis_real_accuracy.png')
                save_discriminator_acc(D_real_acc, D_fake_acc, dis_real_image_path)

            with torch.no_grad():
                fake = netG(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:64], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)

                img_grid_real_np = np.asarray(img_grid_real.detach().cpu())
                img_grid_fake_np = np.asarray(img_grid_fake.detach().cpu())

                if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                    #Save fake images in a grid for every 10th epoch
                    fakeImage = torchvision.utils.make_grid(fake[:64], normalize=True)
                    fakeImage = fakeImage.detach().cpu()
                    imagePathToSave = os.path.join(fake_image_dir, f'dcgan_epoch_{epoch}_img_fake.png')
                    saveImage(fakeImage, imagePathToSave)
                
                # Plot the real images
                plt.figure(figsize=(15,15))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(np.transpose(img_grid_real_np, (1,2,0)))

                # Plot the fake images from the last epoch
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_grid_fake_np,(1,2,0)))
                plt.show()


        if every_100_epochs and (epoch % 100 == 0):
            every_100_epoch_dir = os.path.join(path_save, f'epoch_{epoch}_generated_images_{NUMBER_OF_IMAGES_TO_GENERATE}')
            print(f'Saving 1000 generated images at current epoch {epoch}, in dir {every_100_epoch_dir}')

            try:
                os.mkdir(every_100_epoch_dir)
            except FileExistsError:
                print(f'{every_100_epoch_dir} exist. Overwriting exisiting files.')

            noise = torch.randn(NUMBER_OF_IMAGES_TO_GENERATE, Z_DIM, 1, 1, device=device)
            fake = netG(noise)

            numberOfImagesGenerated = fake.shape[0]

            for imageNumber in range(numberOfImagesGenerated):
                fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
                fakeImage = fakeImage.detach().cpu()
                fakeImage = imageNumpyArray(fakeImage)
                imagePathToSave = os.path.join(every_100_epoch_dir, f'dcgan_img_fake_{imageNumber}.png')
                print(f'saving generated image at {imagePathToSave}')
                saveImage(fakeImage, imagePathToSave)



    """
    Save images generated
    """
    noise = torch.randn(NUMBER_OF_IMAGES_TO_GENERATE, Z_DIM, 1, 1, device=device)
    fake = netG(noise)

    numberOfImagesGenerated = fake.shape[0]

    for imageNumber in range(numberOfImagesGenerated):
        fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
        fakeImage = fakeImage.detach().cpu()
        fakeImage = imageNumpyArray(fakeImage)
        imagePathToSave = os.path.join(new_dir, f'dcgan_img_fake_{imageNumber}.png')
        print(f'saving generated image at {imagePathToSave}')
        saveImage(fakeImage, imagePathToSave)


def generate_dcgan(image_size, path_save, device, num_images_generate):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    """
    Generate fake images using a pre-trained model
    """
    if (device == 'gpu' and not torch.cuda.is_available()):
        print(f'device given {device} but cuda not available. Using cpu to train model.')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = 1 if device == 'cuda' else 0

    if image_size == 128:
        ngf = 128
        ndf = 32
    elif image_size == 64:
        ngf = 64
        ndf = 64

    Z_DIM = 100
    CHANNELS_IMG = 3
    lr = 0.0002
    beta1 = 0.5
    NUMBER_OF_IMAGES_TO_GENERATE = num_images_generate

    # Create the generator
    netG = Generator(ngpu, Z_DIM, CHANNELS_IMG, ngf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu, Z_DIM, CHANNELS_IMG, ndf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print()
    print('Generator summary')
    summary(netG, (Z_DIM, 1, 1))
    print('Discriminator summary')
    summary(netD, (CHANNELS_IMG, image_size, image_size))
    print()

    # load previous model
    modelPath = os.path.join(path_save, 'model.tar')
    if device == 'cuda':
        checkpoint = torch.load(modelPath)
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(modelPath, map_location=device)
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])

    netG.train()
    netD.train()

    new_dir = os.path.join(path_save, f'generated_images')
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        print(f'{new_dir} exist. Overwriting exisiting files.')

    """
    Save images generated
    """
    noise = torch.randn(NUMBER_OF_IMAGES_TO_GENERATE, Z_DIM, 1, 1, device=device)
    fake = netG(noise)

    numberOfImagesGenerated = fake.shape[0]

    for imageNumber in range(numberOfImagesGenerated):
        fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
        fakeImage = fakeImage.detach().cpu()
        fakeImage = imageNumpyArray(fakeImage)
        imagePathToSave = os.path.join(new_dir, f'dcgan_img_fake_{imageNumber}.png')
        print(f'saving generated image at {imagePathToSave}')
        saveImage(fakeImage, imagePathToSave)