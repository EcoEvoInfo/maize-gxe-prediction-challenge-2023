import pandas as pd

# 210,942 imputed SNPs from JC
df_part_A = pd.read_hdf("A_imputed_pandas.h5")

# 7,498 snps associated with adaptation to altitude and latitude
with open('Romero_altlat_NAM5.txt') as f:
    gxe_snps = f.read().splitlines()

# only 209 present in this set
df_gxe = df_part_A.filter(gxe_snps)

# write to file
df_gxe.to_csv('GxE_snps.csv')
