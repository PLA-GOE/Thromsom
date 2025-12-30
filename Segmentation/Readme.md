## Thromsom - Segmentation and value extraction

This tool is part of the github-release for our Paper:

# Making the Invisible Visible: Integrated Visualization and Automated Quantification of Thrombus Deformation During Mechanical Thrombectomy
(Ernst et al., 2026)

For testing purposes and due to githubs size-restraints, we will share a test dataset for each stent-retriever via this link:

[Test-Data and examples](https://owncloud.gwdg.de/index.php/s/l7qJCG250VdI4NJ)

## Prerequisite

Install all requirements from requirements.txt

## Part 1: Segmentation

Start the segmentation with:

    python3 main.py

After loading your image, restrict the area to the vessel as tight as possible (left, right, top bottom), and finally click inside the vessel (middle works most reliable and yields best results).
The algorithm will extract the vessels shape, normalize the image and open the GUI afterwards.

You can now add seedpoints to the thrombus and adjust its threshold. 
We used indexes 1 / green to mark up the thrombus, 2 / yellow for stent-tip, 3 / red for stent-rear.

You can adjust the threshold by selecting the index and clicking into the contrast bar. The click will move the nearest threshold to the clicked value.

Click on near-trace to trace the thrombus over all timestamps.

It is recommended to save Mask and Normalized Scan for later analysis by using the buttons on the top.

Troubleshooting: If the initial vessel segmentation fails due to poor contrast or static thrombi, adjust the parameters in "level_set.py"

## Part 2: Value extraction

Start the value extraction with:

python3 main.py **-stats**

This enables the automated analysis. It requires all former images:

1. The normalized scan
2. The mask
3. The original Dicom.
4. An UID

After a few seconds, it will output the tracing parameters in an SQL'able format in the console. 

## Result

The output of Part 1 will yield a normalized scan in *.nii format as well as the segmentation mask in *.nii format.

Part 2 will output an SQL'able print you can copy and paste into a database to collect all your results.


Contact: mariellesophie.ernst@med.uni-goettingen.de

