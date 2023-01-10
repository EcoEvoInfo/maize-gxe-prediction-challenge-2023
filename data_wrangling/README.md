We created separate feature sets for each 'leave one year out' (LOYO) fold. We pre-processed the component datasets as follows:

- `2_Training_Meta_Data_2014_2021.csv`: we created a single column 'legume' if soybeans or peanuts were planted the previous year.
- `3_Training_Soil_Data_2015_2021.csv`: we dropped columns missing in the majority of fields that had soil data.
- `4_Training_Weather_Data_2014_2021.csv`: we reduced weather data to 1-7 features (depending on LOYO fold) using a 1D-CNN
- `5_Genotype_Data_All_Years.vcf`: we reduced genomic data to 5 latent dimensions, and also included ~200 individual SNPs associated with adaptation to altitude/latitude.
- `6_Training_EC_Data_2014_2021.csv`: we carried out a principal components analysis, and included the first two PCs for each phenophase (18 features)
