## Thromsom - Statistics

This tool is part of the github-release for our Paper:

# Making the Invisible Visible: Integrated Visualization and Automated Quantification of Thrombus Deformation During Mechanical Thrombectomy
(Ernst, 2026)

For testing purposes and due to githubs size-restraints, we will share a test dataset for each stent-retrieber via this link:

[Test-Data and examples](https://owncloud.gwdg.de/index.php/s/l7qJCG250VdI4NJ)

## Usage

Each of the subdirectories in this repository contains a script to generate all of the data presented in the above paper. 

- Segmentation contains the actual tool developed for a robust segmentation
- DefAndConCalc contains a script to calculate the presented values "Thrombus deformation" and "Thrombus contour change" as well as some graphics on the way (Fig. 6)
- GraphGenerator generates image-graph-combinations from the raw data (Fig. 2 - Fig. 4)
- Plotters were used for the violin- and boxplots (Fig. 5, Fig. 7 and Fig. 8)
- Statistics contain multiple scripts used for the statistical calculations in this paper (Tbl. 1 - Tbl. 4)

Instructions are available inside each subdirectory. We recommend you to start at "Segmentation" if you want to recreate the results of this paper.

Please reach out to mariellesophie.ernst@med.uni-goettingen.de if you have any questions regarding the paper.
Contact philip.langer@med.uni-goettingen.de if you have questions regarding the actual code itself.

