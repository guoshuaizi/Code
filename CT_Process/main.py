from resample import resample
from wid import window
from nii2png import nii_to_image
from Image_filtering import DeL




if __name__ == '__main__':

    resample("CT_nii/master","CT_nii/master")
    resample("CT_nii/segmentation","CT_nii/segmentation")


    window("CT_nii/master","CT_nii_window/master","CT_nii/segmentation","CT_nii_window/segmentation")



    mode = ["axial", "coronal", "sagittal"]
    nii_to_image("CT_nii_window/segmentation","CT_2D/Labels",mode[0])
    nii_to_image("CT_nii_window/master", "CT_2D/Images", mode[0])
    nii_to_image("CT_nii_window/segmentation", "CT_2D/Labels", mode[1])
    nii_to_image("CT_nii_window/master", "CT_2D/Images", mode[1])
    nii_to_image("CT_nii_window/segmentation", "CT_2D/Labels", mode[2])
    nii_to_image("CT_nii_window/master", "CT_2D/Images", mode[2])


    DeL("CT_2D/Labels","CT_2D/Images")

