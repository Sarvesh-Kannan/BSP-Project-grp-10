import itertools
import mne
import pyedflib
import pandas as pd

from typing import List, Union, Tuple


def load_signals(
    file_path: str, only_info: bool = False, retrieve_signals: List[str] = None
) -> Union[List, Tuple[List[str], List[float]]]:
    """Load the edf signals for the given filepath."""
    with pyedflib.EdfReader(file_path) as edf:
        start = edf.getStartdatetime()
        signals, frequencies = edf.getSignalLabels(), edf.getSampleFrequencies()
        if only_info:
            return signals, frequencies
        data = []

        for ch_idx, sig_name, freq in zip(
            range(len(signals)),
            signals,
            frequencies,
        ):
            if retrieve_signals is not None and sig_name not in retrieve_signals:
                continue
            sig = edf.readSignal(chn=ch_idx)
            idx = pd.date_range(
                start=start, periods=len(sig), freq=pd.Timedelta(1 / freq, unit="s")
            )
            data.append(pd.Series(sig, index=idx, name=sig_name))
            
    return data



def load_annotations(annotation_file_path: str, psg_file_path: str) -> pd.DataFrame:
    """Load the edf annotations for the given filepath."""
    annotations = mne.read_annotations(annotation_file_path)
    # Fore some hypnogram files there is an error when trying to get the start time
    # => Solution; get the start time from the psg file
    start_time = pyedflib.EdfReader(psg_file_path).getStartdatetime()
    df = pd.DataFrame()

    # Read the onset and set as index
    df["onset"] = annotations.onset
    df["onset"] = start_time + pd.to_timedelta(df["onset"], unit="s")
    df = df.rename(columns={"onset": "start"})
    assert df["start"].is_unique
    df.set_index("start", inplace=True)

    # Read the duration and set as end time
    df["duration"] = annotations.duration
    df["end"] = df.index + pd.to_timedelta(df["duration"], unit="s")
    df.drop(columns="duration", inplace=True)

    # Read the description
    df["description"] = annotations.description

    return df


def annotation_to_30s_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """Convert the annotations to 30 sec labels."""
    if not (annotations.index[1:] == annotations.end[:-1]).all():
        # There is a gap in the annotations
        diffs = (annotations.index[1:] - annotations.end[:-1]).dt.seconds
        gaps = diffs[diffs != 0]
        # The end time of gap_starts indicate the start of the gap
        gap_starts = annotations[:-1][~(annotations.index[1:] == annotations.end[:-1])]
        # The start time of gap_ends indicate the end of the gap
        gap_ends = annotations[1:][~(annotations.index[1:] == annotations.end[:-1])]
        for idx, gap in enumerate(gaps):
            assert gap > 0
            gap_start_label = gap_starts["description"].values[idx]
            gap_end_label = gap_ends["description"].values[idx]
            if gap_start_label == gap_end_label:
                annotations.loc[
                    annotations.index == gap_starts.index[idx], "end"
                ] += pd.Timedelta(gap, unit="s")
            else:
                print("Cannot fix gap")

    index = pd.date_range(
        start=annotations.index[0], end=annotations.end[-1], freq=pd.Timedelta("30s")
    )
    duration = (annotations.end - annotations.index).dt.seconds.values // 30
    labels = itertools.chain.from_iterable(
        [[l] * d for (l, d) in zip(annotations["description"], duration)]
    )
    df = pd.DataFrame({"label": labels}, index=index[:-1])
    df.index.name = "start"

    return df