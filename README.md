# Thromsom-Stats-Tool

This tool is able to trace and automatically segment clots in 3D-printed vessels during angiography image series. 

python3 main.py to start the "default" segmentation. It requires only an image in DICOM format.

For testing purposes and due to githubs size-restraints, we will share a test dataset upon request.

After loading your image, restrict the area to the vessel only, and click inside.
The algorithm will extract the vessels shape, normalize the image and opens the GUI afterwards.

You can now add seedpoints to the thrombus and adjust its threshold. Click on near-trace to trace the thrombus over all timestamps.

It is recommended to save Mask and Normalized Scan for later analysis.


python3 main.py -stats enables the automated analysis. It required all former images:
1. The normalized scan
2. The mask
3. The original Dicom.
4. An UID

After a few seconds, it will output the tracing parameters in an SQL'able format. 

Further analysis-tools are available upon request.

Contact: mariellesophie.ernst@med.uni-goettingen.de