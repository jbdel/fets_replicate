import nibabel as nib

import numpy as np

proxy_img = nib.load('Brats18_2013_0_1_seg.nii.gz')
proxy_img.uncache()
img = np.array(proxy_img.dataobj)
print(img.shape)

regions = {"ET_brats": 4, "ET": 3,  "NCR-NET": 1, "ED": 2}
img[img == regions["ET_brats"]] = regions["ET"]

print(img)
