from Utils import get_dataloader
from TrainWgan import train_wgan, generate_wgan
from TrainDcgan import train_dcgan, generate_dcgan
import os

if __name__ == "__main__":
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train gan model')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')

    parser.add_argument('--image_size', required=False,
                        default = 64,
                        metavar="image size of input image",
                        help='The height x width of image to train the gan model')

    parser.add_argument('--batch_size', required=False,
                        default = 128,
                        metavar="Batch size",
                        help='Batch size')

    parser.add_argument('--epochs', required=False,
                        default = 500,
                        metavar="number of epochs",
                        help='How many epochs to train')

    parser.add_argument('--device', required=False,
                        default = 'cpu',
                        metavar="cpu or gpu to train",
                        help='What device to train the model')

    parser.add_argument('--path_save', required=True,
                        metavar="Path to save image generated",
                        help='Where to save the image generated')

    parser.add_argument('--gan_type', required=False,
                        default = 'dcgan',
                        metavar="DCGAN or Wgan",
                        help='What gans to use')

    parser.add_argument('--resume_training', required=False,
                        default = False,
                        metavar="Resume training]",
                        help='Resume training from model')

    parser.add_argument('--save_model', required=False,
                        default = True,
                        metavar="Save model",
                        help='Save model on every epoch')

    parser.add_argument('--generate_fake', required=False,
                        default = False,
                        metavar="Generate 1000 fake images",
                        help='Generate 1000 fake images')

    parser.add_argument('--every_100_epochs', required=False,
                        default = True,
                        metavar="Save images for every 100 epoch",
                        help="Save images for every 100 epoch")

    parser.add_argument('--num_images_generate', required=False,
                        default = 1000,
                        metavar="Generate 1000 images",
                        help="Generate 1000 images")

    args = parser.parse_args()

    DATAROOT = args.dataset
    IMAGE_SIZE = int(args.image_size)
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    DEVICE = args.device
    GAN_TYPE = args.gan_type
    PATH_SAVE = args.path_save
    RESUME_TRAINING = args.resume_training
    SAVE_TRAINING = args.save_model
    GENERATE_FAKE = args.generate_fake
    EVERY_100_EPOCH = args.every_100_epochs
    NUM_IMAGES_GENERATE = int(args.num_images_generate)
    num_workers = 2

    print(f'Arguments given')
    print(f'Dataset {DATAROOT}')
    dataset_dir = os.listdir(DATAROOT)
    dataset_subdir = dataset_dir[0]
    dataset_dir = os.path.join(DATAROOT, dataset_subdir)
    num_images_for_training = len(os.listdir(dataset_dir))
    print(f'Using dataset of size {num_images_for_training} to train')

    print(f'Image size {IMAGE_SIZE}')
    if (IMAGE_SIZE != 64 or IMAGE_SIZE != 128):
        print(f'Image size given is not 64 nor 128. Will use default image size as 64.')
        IMAGE_SIZE = 64
    
    print(f'Batch size {BATCH_SIZE}')
    if (BATCH_SIZE is None):
        print(f'Batch size not provided, will use default batch size as 128.')

    print(f'Epochs {EPOCHS}')
    if (EPOCHS is None):
        print(f'Epochs not provided, will use default epoch as 500.')
    
    print(f'Gan Type {GAN_TYPE}')
    if (GAN_TYPE != 'dcgan' or GAN_TYPE != 'wgan'):
        print(f'Gan type provided is incorrect. Will use default dcgan as the GAN type.')
        GAN_TYPE = 'dcgan'
    
    print(f'Save images generated every 100 epochs {EVERY_100_EPOCH}')
    print()

    """
    Get data loader
    """
    loader = get_dataloader(DATAROOT, IMAGE_SIZE, BATCH_SIZE, num_workers)

    """
    Training of WGAN-GP
    """
    if GAN_TYPE == 'wgan' and not GENERATE_FAKE:
        print(f'Starting training with {GAN_TYPE}')
        train_wgan(IMAGE_SIZE, BATCH_SIZE, EPOCHS, DEVICE, loader, PATH_SAVE, RESUME_TRAINING, SAVE_TRAINING, EVERY_100_EPOCH, NUM_IMAGES_GENERATE)
    elif GAN_TYPE == 'dcgan' and not GENERATE_FAKE:
        print(f'Starting training with {GAN_TYPE}')
        train_dcgan(IMAGE_SIZE, BATCH_SIZE, EPOCHS, DEVICE, loader, PATH_SAVE, RESUME_TRAINING, SAVE_TRAINING, EVERY_100_EPOCH, NUM_IMAGES_GENERATE)

    if GENERATE_FAKE and GAN_TYPE == 'wgan':
        print(f'Generating fake images with {GAN_TYPE}')
        generate_wgan(IMAGE_SIZE, PATH_SAVE, DEVICE, NUM_IMAGES_GENERATE)
    elif GENERATE_FAKE and GAN_TYPE == 'dcgan':
        print(f'Generating fake images with {GAN_TYPE}')
        generate_dcgan(IMAGE_SIZE, PATH_SAVE, DEVICE, NUM_IMAGES_GENERATE)
