from model import D, G
import torch
import matplotlib.image as mpimg
import numpy as np
import sys
from argparse import ArgumentParser

check_gpu = torch.cuda.is_available()
weight_path ='weights/G.pth'

G.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
if check_gpu is True:
    G.load_state_dict(torch.load(weight_path),map_location=torch.device('cuda'))

def generate_images(no_of_images):
    for i in range(int(no_of_images)):
        if check_gpu is True:
            img = G(torch.from_numpy(np.random.uniform(-1, 1, size=(1,100))).cuda().float()).squeeze_(0).permute(1,2,0).detach().cpu().numpy()
        else:
            img = G(torch.from_numpy(np.random.uniform(-1, 1, size=(1,100))).float()).squeeze_(0).permute(1,2,0).detach().cpu().numpy()
        img_scaled = ((img + 1)*255 / (2)).astype(np.uint8)
        print("Saved img_{}.jpg".format(i))
        mpimg.imsave('generated_samples/img_{}.jpg'.format(i), img_scaled)
    print("Successfully saved {} images to folder generated_samples".format(no_of_images))


def argument():
	parser = ArgumentParser("Download the AI generated 28x28 pixel images\n")
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