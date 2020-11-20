from model import D, G
from tqdm.autonotebook import tqdm
from torchvision.utils import make_grid
import torch
import imageio
import numpy as np
from argparse import ArgumentParser

check_gpu = torch.cuda.is_available()
weight_path = 'weights/G.pth'

G.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
if check_gpu is True:
    G.load_state_dict(torch.load(weight_path), map_location=torch.device('cuda'))


def generate_images(no_of_images, grid_size=10):
    for i in tqdm(range(no_of_images)):
        with torch.no_grad():
            imgs = []
            for _ in range(grid_size):
                img = G(torch.from_numpy(np.random.uniform(-1, 1, size=(1, 100))).float()).squeeze_(0).numpy()
                imgs.append(img)
            grid = make_grid(torch.as_tensor(imgs), nrow=5).permute(1, 2, 0).numpy()
            grid = np.array((grid - np.min(grid)) / (np.max(grid) - np.min(grid)))  # :D without normalizing looks cool
            imageio.imwrite('generated_samples/img_{}.jpg'.format(i), grid)
    print("Successfully saved {} images to folder generated_samples".format(no_of_images))


def argument():
    parser = ArgumentParser("Download the AI generated 32x32 pixel images\n")
    n_help = "number of images to save"
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument("-n", help=n_help, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument()
    generate_images(args.n)


if __name__ == '__main__':
    main()
