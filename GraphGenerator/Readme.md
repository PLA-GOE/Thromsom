## Thromsom - Graph Generator

This tool is part of the github-release for our Paper:

# Making the Invisible Visible: Integrated Visualization and Automated Quantification of Thrombus Deformation During Mechanical Thrombectomy
(Ernst et al., 2026)

For testing purposes and due to githubs size-restraints, we will share a test dataset for each stent-retriever via this link:

[Test-Data and examples](https://owncloud.gwdg.de/index.php/s/l7qJCG250VdI4NJ)

## Prerequisite

Install all requirements from the requirements.txt

The resulting data.csv from Thromsom/DefAndConCalc is required.
The resulting mask from Thromsom/Segmentation is required.
You have to convert the original DICOM-Series to *.nii beforehand (f.e. [3D-Slicer](https://www.slicer.org/))

Adjust the lines

    data_path = "PATH TO data.csv FROM DEF AND CON CALC"
    mask_path = "PATH TO mask.nii FROM SEGMENTATION"
    scan_path = "PATH TO scan.nii (CONVERTED SCAN FROM ORIGINAL DICOM)"

accordingly.

## Part 1

To generate the image series in combination with the Graphs, run the script with

    python3 main.py

## Result

This script will net a series of images for each slice of the thrombectomy.

Figure 2, figure 3 and figure 4 from our paper were generates this way.

You can use [ffmpeg](https://www.ffmpeg.org) to combine the images to your liking.

Exemplary animations:

[Successful extraction](https://owncloud.gwdg.de/index.php/s/xU55iKa6LBGj3ly)

[Shortening](https://owncloud.gwdg.de/index.php/s/fAVhi6P4dLFpKYV)

[Failed extraction](https://owncloud.gwdg.de/index.php/s/y2LrJwGboZJQYV9)


Contact: mariellesophie.ernst@med.uni-goettingen.de

