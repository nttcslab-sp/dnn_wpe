import argparse
from pathlib import Path
from typing import List

import h5py
from typeguard import check_argument_types


def main(in_h5: List[str], out_h5: str):
    assert check_argument_types()
    try:
        with h5py.File(out_h5, 'w') as fo:
            for h in in_h5:
                with h5py.File(h, 'r') as fi:
                    for k in fi:
                        fo.create_dataset(k, data=fi[k][()],
                                          compression='gzip')
    except:
        if Path(out_h5).exists():
            Path(out_h5).unlink()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-o', '--out-h5', required=True)
    args = parser.parse_args()

    main(in_h5=args.inputs, out_h5=args.out_h5)
