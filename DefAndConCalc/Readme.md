## Thromsom - Deformation and ContourChange Calculator

This tool is part of the github-release for our Paper:

# Making the Invisible Visible: Integrated Visualization and Automated Quantification of Thrombus Deformation During Mechanical Thrombectomy
(Ernst et al., 2026)

For testing purposes and due to githubs size-restraints, we will share a test dataset for each stent-retriever via this link:

[Test-Data and examples](https://owncloud.gwdg.de/index.php/s/l7qJCG250VdI4NJ)

## Prerequisite

Install all requirements from the requirements.txt

The resulting mask from Thromsom/Segmentation is required.

Adjust the line

    dir = "INSERT DIR TO SEGMENTATIONS HERE"

in main.py to point to your folder with your segmentation(s).

## Part 1

To extract the deformation and contour change parameters, start the script with

    python3 main.py

## Result

The resulting data will yield a directory containing data.csv, eval.png and multiple *.nii for each NIfTI in the input directory.

Data.csv contains all information for deformation and contour-change, slice-wise calculated.

Eval.png equals Figure 6 from our paper.


Contact: mariellesophie.ernst@med.uni-goettingen.de

