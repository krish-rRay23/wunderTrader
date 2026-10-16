"""Fast sequence reader for Parquet sequences without requiring end-of-file metadata."""
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
import fastparquet.cencoding as ce
import cramjam

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH

COLUMN_NAMES = ['seq_ix', 'step_in_seq', 'need_prediction'] + FEATURE_COLUMNS + list(TARGET_COLUMNS)
NUM_COLUMNS = len(COLUMN_NAMES)


class FastSequenceReader:
    def __init__(self, file_path: str | Path = 'datasets/train.parquet'):
        self.file_path = Path(file_path)

    def iter_sequences(self, max_sequences: int | None = None) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Yields (seq_ix, features, targets) for each complete 20,000-row sequence.
        features: np.ndarray of shape (20000, 112), dtype float32
        targets: np.ndarray of shape (20000, 2), dtype float32
        """
        with open(self.file_path, 'rb') as f:
            f.seek(4) # Skip PAR1
            seq_idx = 0
            while max_sequences is None or seq_idx < max_sequences:
                seq_start_pos = f.tell()
                seq_id = None
                features = np.empty((SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
                targets = np.empty((SEQUENCE_LENGTH, 2), dtype=np.float32)
                feat_col_idx = 0

                try:
                    for col_name in COLUMN_NAMES:
                        cur = f.tell()
                        buf = f.read(2000)
                        if len(buf) < 10:
                            return
                        nio = ce.NumpyIO(np.frombuffer(buf, dtype=np.uint8))
                        header = ce.read_thrift(nio)
                        hlen = nio.tell()
                        comp_size = header[3]

                        f.seek(cur + hlen)
                        comp_data = f.read(comp_size)
                        pos_zstd = comp_data.find(b'(\xb5/\xfd')
                        decomp = cramjam.zstd.decompress(comp_data[pos_zstd:])

                        if col_name == 'seq_ix':
                            arr = np.frombuffer(decomp, dtype=np.int32)
                            seq_id = int(arr[0])
                        elif col_name in ('step_in_seq', 'need_prediction'):
                            pass
                        elif col_name in ('t0', 't1'):
                            # float32 with BYTE_STREAM_SPLIT (4 streams of 20000 bytes)
                            raw = np.frombuffer(decomp, dtype=np.uint8).reshape(4, SEQUENCE_LENGTH)
                            unsplit = raw.T.copy().view(np.float32).reshape(SEQUENCE_LENGTH)
                            if col_name == 't0':
                                targets[:, 0] = unsplit
                            else:
                                targets[:, 1] = unsplit
                        else:
                            # Feature columns: float32 BYTE_STREAM_SPLIT
                            raw = np.frombuffer(decomp, dtype=np.uint8).reshape(4, SEQUENCE_LENGTH)
                            unsplit = raw.T.copy().view(np.float32).reshape(SEQUENCE_LENGTH)
                            features[:, feat_col_idx] = unsplit
                            feat_col_idx += 1

                        f.seek(cur + hlen + comp_size)

                    yield seq_id, features, targets
                    seq_idx += 1

                except Exception:
                    # End of file or incomplete row group
                    return
