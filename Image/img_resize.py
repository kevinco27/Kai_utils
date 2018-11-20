import numpy as np
import scipy.misc
import os


def resize(inDir_path, outDir_path, w_resize):
	file_list = os.listdir(inDir_path)
	for file in file_list:
		*file_name, file_format = file.split('.')
		image = scipy.misc.imread(inDir_path+'\\'+file)
		width = image.shape[1]
		scale = w_resize / width
		resize_img = scipy.misc.imresize(image, scale)
		scipy.misc.imsave(outDir_path+'\\'+file_name[0]+'_w190.png', resize_img)