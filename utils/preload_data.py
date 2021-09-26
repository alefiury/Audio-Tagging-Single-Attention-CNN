import logging

import librosa
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset

# Logger
log = logging.getLogger(__name__)

def audio_file_to_array(batch: datasets.arrow_dataset.Dataset,
                            data_path: str,
                            class_num: int,
                            sample_rate: int) -> datasets.arrow_dataset.Dataset:
    """
    Loads the audios from memory.

    The audios are loaded from disk and saved in a format that will speed up training afterwards.

    ----
    Args:
        batch:
            A huggingface datasets element.
        data_path:
            Path to audio files.
        class_num:
            Number of classes.
        sample_rate:
            Desired sample rate.

    Returns:
        A huggingface datasets element.
    """

    y, sr = librosa.load(f"{data_path}/{batch['filename']}",
                            sr=sample_rate,
                            mono=True,
                            res_type='kaiser_fast')

    batch['image'] = y
    batch['sample_rate'] = sr

    # Creates a hot one encoding version of the label especie_id
    label = np.zeros(class_num, dtype='f')
    label[batch['target']] = 1
    batch['hot_target'] = label

    return batch


def prepare_data(df: pd.core.frame.DataFrame,
                    data_path: str,
                    class_num: int,
                    sample_rate: int,
                    num_workers: int) -> datasets.arrow_dataset.Dataset:
    """
    Preloads the audio files.

    The audio files are loaded and saved in disk to accelerate training.

    ----
    Args:
        preloaded_data_path:
            Path to the directory where the preloaded dataset will be saved.
        data_path:
            Path to audio files.
        class_num:
            Number of classes.
        sample_rate:
            Desired sample rate.
        num_workers:
            Number of processes (Multiprocessing).

    Returns:
        Huggingface datasets
    """

    log.info('Loading Audios... ')

    # Imports a pandas dataframe to a huggingface dataset
    datasets_df = Dataset.from_pandas(df)

    dataset = datasets_df.map(audio_file_to_array,
                                    fn_kwargs={"data_path": data_path,
                                        "class_num": class_num,
                                        "sample_rate": sample_rate},
                                    num_proc=num_workers)

    return dataset