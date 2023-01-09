The maize GxE [competition
data](https://drive.google.com/drive/folders/1leYJY4bA3341S-JxjBIgmmAWMwVDHYRb)
includes genotype data for 4,928 individuals at 437,214 variant sites.

To reduce the number of genetic features in this dataset, we used two strategies. First, we captured genome-wide relationships using a variational autoencoder method developed by [@battey2021]. Our VAE was trained to reduce all genotype data into 5 latent dimensions.

Second, we leveraged previous knowledge of maize evolutionarily history to identify a subset of SNPs
important for adaptation to diverse environments. Specifically, we
selected a subset of X SNPs that were previously implicated in
adaptation of 4,471 maize landraces from across Mexico, Central America,
and South America to altitude and latitude [@romeronavarro2017]. The majority of
SNPs associated with altitude (61%) were also associated with variation in flowering time
in experiments [@romeronavarro2017]. Early flowering is an important
adaptation to high latitudes and high elevations, where there is a
shorter growing season. We hypothesized that SNPs predictive of flowering time variation
could be important for explaining variation in performance of different
individuals across different environments in the G2F locations.

Genotype data for the landraces studied by @romeronavarro2017 were
downloaded from the CIMMYT Seeds of Discovery project [@hearne2012].
