# Verify that the refactors still give the same output

import filecmp
print("The two refactored versions produce the same output:")
print(filecmp.cmp(
        r"H_Backbone_tracing/misc_outputs\refactor1_agglomerative_cd_4.0_CA_only.pdb", 
        r"H_Backbone_tracing/misc_outputs\refactor2_agglomerative_cd_4.0_CA_only.pdb"
    )
)
