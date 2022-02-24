import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchsummary import summary
from ModelWgan import Generator, Discriminator, initialize_weights
from Utils import saveImage, imageNumpyArray, save_gen_and_critic, get_dataloader, gradient_penalty
import random

def train_wgan(image_size, batch_size, epochs, device, loader, path_save, resume_training, save_model, every_100_epochs, num_images_generate):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    if (device == 'gpu' and not torch.cuda.is_available()):
        print(f'device given {device} but cuda not available. Using cpu to train model.')

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        FEATURES_GEN = 128
        FEATURES_CRITIC = 32
    elif image_size == 64:
        FEATURES_GEN = 64
        FEATURES_CRITIC = 64

    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)

    if resume_training:
        modelPath = os.path.join(path_save, 'model.tar')
        checkpoint = torch.load(modelPath)

        gen.load_state_dict(checkpoint['generator'])
        critic.load_state_dict(checkpoint['discriminator'])
        opt_gen.load_state_dict(checkpoint['optimizerG'])
        opt_critic.load_state_dict(checkpoint['optimizerD'])

        gen.train()
        critic.train()

    gen.train()
    critic.train()

    # gen loss
    gen_loss_list = list()
    critic_loss_list = list()

    print()
    print('Generator summary')
    summary(gen, (Z_DIM, 1, 1))
    print('Critic summary')
    summary(critic, (CHANNELS_IMG, image_size, image_size))
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

    new_dir = os.path.join(path_save, f'generated_images')
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        print(f'{new_dir} exist. Overwriting exisiting files.')

    print("Starting Training Loop...")
    for epoch in range(1, NUM_EPOCHS+1):
        print(f'epoch {epoch}')
        loss_critic = ''

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]
        
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            print(f'Training critic...')
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # add gen loss and critic loss to list
            gen_loss_list.append(loss_gen.item())
            critic_loss_list.append(loss_critic.item())

            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], batch_idx {batch_idx} | {len(loader)} | Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

        """
        Saving model at current epoch 
        """
        if save_model:
            modelPathSave = os.path.join(path_save, 'model.tar')
            torch.save({
                'generator': gen.state_dict(),
                'discriminator': critic.state_dict(),
                'optimizerG': opt_gen.state_dict(),
                'optimizerD': opt_critic.state_dict(),
                }, modelPathSave)
        
        
        # Print losses occasionally
        if epoch >= 0:

            if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                # Save gen and critic loss plot
                gen_and_critic_image_path = os.path.join(loss_plot_dir, f'epoch_{epoch}_gen_critic_loss.png')
                save_gen_and_critic(gen_loss_list, critic_loss_list, 'wgan', gen_and_critic_image_path)


            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:64], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)

                img_grid_real_np = np.asarray(img_grid_real.detach().cpu())
                img_grid_fake_np = np.asarray(img_grid_fake.detach().cpu())

                if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                    #Save fake images in a grid for every 10th epoch
                    fakeImage = torchvision.utils.make_grid(fake[:64], normalize=True)
                    fakeImage = fakeImage.detach().cpu()
                    imagePathToSave = os.path.join(fake_image_dir, f'wgan_epoch_{epoch}_img_fake.png')
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
            fake = gen(noise)

            numberOfImagesGenerated = fake.shape[0]

            for imageNumber in range(numberOfImagesGenerated):
                fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
                fakeImage = fakeImage.detach().cpu()
                fakeImage = imageNumpyArray(fakeImage)
                imagePathToSave = os.path.join(every_100_epoch_dir, f'wgan_img_fake_{imageNumber}.png')
                print(f'saving generated image at {imagePathToSave}')
                saveImage(fakeImage, imagePathToSave)

    """
    Save images generated
    """
    noise = torch.randn(NUMBER_OF_IMAGES_TO_GENERATE, Z_DIM, 1, 1, device=device)
    fake = gen(noise)

    numberOfImagesGenerated = fake.shape[0]

    for imageNumber in range(numberOfImagesGenerated):
        fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
        fakeImage = fakeImage.detach().cpu()
        fakeImage = imageNumpyArray(fakeImage)
        imagePathToSave = os.path.join(new_dir, f'wgan_img_fake_{imageNumber}.png')
        print(f'saving generated image at {imagePathToSave}')
        saveImage(fakeImage, imagePathToSave)


def generate_wgan(image_size, path_save, device, num_images_generate):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    """
    Generate fake images using a pre-trained model
    """
    if (device == 'gpu' and not torch.cuda.is_available()):
        print(f'device given {device} but cuda not available. Using cpu to train model.')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if image_size == 128:
        FEATURES_GEN = 128
        FEATURES_CRITIC = 32
    elif image_size == 64:
        FEATURES_GEN = 64
        FEATURES_CRITIC = 64

    Z_DIM = 100
    CHANNELS_IMG = 3
    LEARNING_RATE = 1e-4
    NUMBER_OF_IMAGES_TO_GENERATE = num_images_generate

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    print()
    print('Generator summary')
    summary(gen, (Z_DIM, 1, 1))
    print('Critic summary')
    summary(critic, (CHANNELS_IMG, image_size, image_size))
    print()

    # load previous model
    modelPath = os.path.join(path_save, 'model.tar')
    if device == 'cuda':
        checkpoint = torch.load(modelPath)
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(modelPath, map_location=device)
    gen.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['discriminator'])
    opt_gen.load_state_dict(checkpoint['optimizerG'])
    opt_critic.load_state_dict(checkpoint['optimizerD'])

    gen.train()
    critic.train()

    new_dir = os.path.join(path_save, f'generated_images')
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        print(f'{new_dir} exist. Overwriting exisiting files.')

    """
    Save images generated
    """ 
    noise = torch.randn(NUMBER_OF_IMAGES_TO_GENERATE, Z_DIM, 1, 1, device=device)
    fake = gen(noise)

    numberOfImagesGenerated = fake.shape[0]

    for imageNumber in range(numberOfImagesGenerated):
        fakeImage = torchvision.utils.make_grid(fake[imageNumber], normalize=True)
        fakeImage = fakeImage.detach().cpu()
        fakeImage = imageNumpyArray(fakeImage)
        imagePathToSave = os.path.join(new_dir, f'wgan_img_fake_{imageNumber}.png')
        print(f'saving generated image at {imagePathToSave}')
        saveImage(fakeImage, imagePathToSave)