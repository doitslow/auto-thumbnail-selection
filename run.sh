#!/bin/bash
cd /raid/P15/2-code/thumbnail
#DIN=/raid/P15/4-data/test
DIN=/raid/P15/4-data/test/The_Wish/DAP22807_CU

ARGS=(
"--din $DIN"
"--match_batch 32" # consider increase to a much larger value
"--caption_batch 2" # consider increase to a much larger value
"--clean_batch 3" # consider increase to a much larger value
"--cleaning c2"
"--meta /raid/P15/4-data/Metadata.xlsx"
"--post_hecate"
)

#python pipeline.py -m m1 ${ARGS[@]}
#python pipeline.py -m m2 ${ARGS[@]}
python content_pipe.py -m en ${ARGS[@]}