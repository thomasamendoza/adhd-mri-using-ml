# Let's load some other packages we need
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib # common way of importing nibabel

mri_file = 'sub-01_task-auditory_bold.nii'
img = nib.load(mri_file)

print(type(img))
print("dimensions in voxels", img.shape) # dimensions in voxels
print("voxel size", img.header.get_zooms()) # tells us voxel size
print("units of voxels and time", img.header.get_xyzt_units()) # units of time and voxels


img_data = img.get_fdata()
print(type(img_data))  # it's a numpy array!
print(img_data.shape)
mid_slice_x = img_data[32, :, :]
print(mid_slice_x.shape)

plt.imshow(mid_slice_x.T, cmap='gray', origin='lower')
plt.xlabel('First axis')
plt.ylabel('Second axis')
plt.colorbar(label='Signal intensity')
plt.show()
