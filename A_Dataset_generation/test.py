import json
import os

import numpy as np

web_api = "https://www.ebi.ac.uk/emdb/api/"
search_filter = "search/" + 'rna AND sample_type:"complex" AND resolution:[1 TO 3} AND release_date:[2023-01-01T00:00:00Z TO 2023-12-31T23:59:59Z] AND xref_links:"pdb"'
download_filter = "?wt=csv&download=true&fl=emdb_id,title,resolution,fitted_pdbs,map_release_date,overall_molecular_weight,xref_ALPHAFOLD"

print(web_api+search_filter+download_filter)