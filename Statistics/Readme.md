## Thromsom - Statistics

This tool is part of the github-release for our Paper:

# Making the Invisible Visible: Integrated Visualization and Automated Quantification of Thrombus Deformation During Mechanical Thrombectomy
(Ernst, 2026)

For testing purposes and due to githubs size-restraints, we will share a test dataset for each stent-retrieber via this link:

[Test-Data and examples](https://owncloud.gwdg.de/index.php/s/l7qJCG250VdI4NJ)

## Prerequisite

Install all requirements from requirements.txt

## Part 1: Statistics for most tables

In this part of the project, you will find all Python scripts used to create the tables in the above publication.

You have to fill the empty arrays accordingly with your values and run each script with

python3 ...

mwu.py runs a mann-whitney-unit-test / wilcoxon-rank-sum-test

CI.py calculates the confidence intervals

fdr.py adjusts the results of mwu.py for multiple outcomes

regression.py calculates the influence of clot-length on deformationa and contour change

interrater.py was used to determine our inter- and intrarater performance

Most values in CAPSLOCK have to be adjusted to **your** path/values/parameters.


## Result

The resulting numerical values from all scripts are the ones appearing in the tables.

Values presented in the paper but not present in the results from these scripts (f.e. mean, range, SD) were calculated with Excels integrated formulas.


Contact: mariellesophie.ernst@med.uni-goettingen.de

