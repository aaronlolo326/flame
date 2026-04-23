import os
from multiprocessing import Pool
from pprint import pprint

from streaming import MDSWriter, StreamingDataset

data_dir = "/storage/backup/hei/data"
mds_subsets = [
    "thestackv1_concat_by_repo-524288",
    "thestackv1_concat_by_repo-65536",
    "book-524288",
    "book-65536",
    "fineweb-edu",
    "fineweb-2023-50",
    "stackexchange",
    "dolmawiki",
    "tuluv2",
    "arxiv",
    "openwebmath",
    "textbooks"
]
mds_dirs = [os.path.join(data_dir, subset) for subset in mds_subsets]
seq_len = 131072

# for each dataset, load the mds file and print the first row
for dir in mds_dirs:
    dataset = StreamingDataset(local=dir,
                               remote=None,
                               shuffle=False,
                               batch_size=1,)
    for sample in dataset:
        pprint(sample)
        breakpoint()
        break
        '''
        a sample looks like this:
            {'domain': '.',
            'indices': array([[     0, 383367],
                [383367, 524288]], dtype=uint32),
            'input_ids': array([128000,  39512,   3249, ...,   2720,    315,  79401], dtype=uint32),
            'length': 524288}
        '''




