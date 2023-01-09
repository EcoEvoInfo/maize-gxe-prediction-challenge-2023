## Overview 
The maize GxE [competition
data](https://drive.google.com/drive/folders/1leYJY4bA3341S-JxjBIgmmAWMwVDHYRb)
includes genotype data for 4,928 individuals at 437,214 variant sites.

To reduce the number of genetic features in this dataset, we used two strategies. First, we captured genome-wide relationships using a variational autoencoder method developed by [Battey et al. 2021](https://academic.oup.com/g3journal/article/11/1/jkaa036/6105578). Our VAE was trained to reduce all genotype data into 5 latent dimensions. Second, we leveraged previous knowledge of maize evolutionarily history to identify a subset of SNPs
important for adaptation to diverse environments. Specifically, we
selected a subset of SNPs that were previously implicated in
adaptation of 4,471 maize landraces from across Mexico, Central America,
and South America to altitude and latitude [(Romero Navarro et al. 2017)](https://doi.org/10.1038/ng.3784). The majority of
SNPs associated with altitude (61%) were also associated with variation in flowering time
in experiments [(Romero Navarro et al. 2017)](https://doi.org/10.1038/ng.3784). Early flowering is an important
adaptation to high latitudes and high elevations, where there is a
shorter growing season. We hypothesized that SNPs predictive of flowering time variation ('GxE SNPs')
could be important for explaining variation in performance of different
individuals across latitudes in the G2F locations. 

## popVAE latent dimensions

## GxE SNPs
We downloadeded a list of SNPs present in Supplemental Table 6 of [(Romero Navarro et al. 2017)](https://doi.org/10.1038/ng.3784). These were converted to BED format and lifted over to using the CrossMap online tool at [EnsemblPlants](https://plants.ensembl.org/Zea_mays/Tools/AssemblyConverter). SNPs were lifted over from coordinates in the *Z. mays* AGPv2 reference to B73_RefGen_v4 and then to Zm-B73-REFERENCE-NAM-5.0. Of 7498 sites in Table S6, only 338 were identified in `5_Genotype_Data_All_Years.vcf` file released with the competition data after filtering. 
