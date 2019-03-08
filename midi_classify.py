from multiprocessing import Pool, cpu_count

from glob import glob
from pickle import dump, HIGHEST_PROTOCOL

from itertools import permutations

from music21 import *
from music21.stream import Score


"""     Info:
            Import albums and songs to sound sequences with labels
            usage: arrange sample files as /samples/<album_name>/<song_name>.mid
"""


MAX_OCTAVE = 7
MAX_SUSTAIN = 8     # x 16th notes.
MAX_VELOCITY = 127


note_dict = {
    'A' : 0,
    'A#': 1, 'B-': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4, 'D-': 4,
    'D' : 5,
    'D#': 6, 'E-': 6,
    'E' : 7,
    'F' : 8,
    'F#': 9, 'G-': 9,
    'G' :10,
    'G#':11, 'A-': 11,
    'R' :12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10:'G',
    11:'G#',
    12:'R'
}

empty_vec = [0 for _ in range(len(note_reverse_dict))]


def preproc_raw_file(raw_file):
    # try:
        sample = converter.parse(raw_file)
        parts = analyze_parts(sample)

        parts_preproced = []
        for part in parts:
            part_preproced = []
            for element in part.flat.elements:
                element_preproced = vectorize_element(element)
                if element_preproced is not None:
                    part_preproced.append(element_preproced)
            if len(part_preproced) != 0:
                parts_preproced.extend(split_part(part_preproced))

        part_preproced = []
        for element in sample.flat.elements:
            vector = vectorize_element(element)
            if vector:
                part_preproced.append(vector)
        if len(part_preproced):
            parts_preproced.extend(split_part(part_preproced))

        return parts_preproced

    # except Exception as e: print(f"! Bad file: {raw_file}: {e}") if verbose else None


def analyze_parts(sample):
    # try:
        all_parts = instrument.partitionByInstrument(sample)
        return [(Score(parts)) for parts in permutations(all_parts)]
    # except Exception as e: print(f"! Bad file: {filename} {e}") if verbose else []


def vectorize_element(element):
    vocab_vect = empty_vec.copy()
    oct_vect   = empty_vec.copy()
    dur_vect   = empty_vec.copy()
    vol_vect   = empty_vec.copy()

    try:
        if element.isNote:
            note_id = note_dict[element.pitch.name]
            if hasValid_duration(element):
                vocab_vect[note_id] += 1
                oct_vect[note_id] += float(element.pitch.octave)
                dur_vect[note_id] += float(element.duration.quarterLength)
                vol_vect[note_id] += float(element.volume.velocity)

        elif element.isChord:
            for e in element:
                note_id = note_dict[e.pitch.name]
                if hasValid_duration(e):
                    duplicateNote = vocab_vect[note_id] != 0
                    vocab_vect[note_id] += 1
                    oct_vect[note_id] += float(e.pitch.octave)
                    dur_vect[note_id] += float(e.duration.quarterLength)
                    vol_vect[note_id] += float(e.volume.velocity)

                    if duplicateNote:
                        oct_vect[note_id] /= 2
                        dur_vect[note_id] /= 2
                        vol_vect[note_id] /= 2

        elif element.isRest:
            if hasValid_duration(element):
                note_id = note_dict['R']
                vocab_vect[note_id] += 1
                dur_vect[note_id] += float(element.duration.quarterLength)

    # normalization & final-fixes

        vocab_sum = sum(vocab_vect)

        if vocab_sum == 0: return None

        if vocab_sum != 1: vocab_vect = [round(float(e / vocab_sum), 3) for e in vocab_vect]
        oct_vect = [e / MAX_OCTAVE for e in oct_vect]
        dur_vect = [e / MAX_SUSTAIN for e in dur_vect]
        vol_vect = [e / MAX_VELOCITY for e in vol_vect]

    except Exception as e:
        # if verbose: print(f'Catch : Element {element} : {e}')
        return None

    vect = []
    vect.extend(vocab_vect)
    vect.extend(oct_vect)
    vect.extend(dur_vect)
    vect.extend(vol_vect)
    return vect


def split_part(part_preproced):
    samples = []

    container = []
    for element in part_preproced:
        durations = element[int(len(element)*1/2):int(len(element)*3/4)]
        if max(durations) > MAX_SUSTAIN*1/2:
            samples.append(container)
            container = []
        else:
            container.append(element)

    if container != []: samples.append(container)

    return samples


def hasValid_duration(element): return 0.0 < float(element.duration.quarterLength) <= MAX_SUSTAIN





def pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        return dump(obj, MacOSFile(f), protocol=HIGHEST_PROTOCOL)

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size





def class_pars_fn(args):
    i, folder = args
    samples = glob(folder+"*.mid")
    class_data = []
    label = [0 if _ != i else 1 for _ in range(hm_classes)]
    if verbose: print(f"class {i+1} : {folder} {len(samples)} files.")
    for raw_file in samples:
        # if verbose: print(f"> working on: {raw_file}")
        samples = preproc_raw_file(raw_file)
        if samples:
            for sample in samples:
                class_data.append([sample, label])


    pickle_save(class_data, f"class{i+1}.pkl")
    with open(f'class{i+1}.txt', 'w+') as f:
        for (seq,lbl) in class_data:
            for e in lbl:
                f.write(str(e) + " ")
            f.write("\n")
            for t in seq:
                for e in t:
                    f.write(str(e) + " ")
                f.write("\n")
            f.write("\n;\n")
    if verbose: print(f"class {i+1}: obtained {len(class_data)} samples.")


verbose = True
sample_folders = glob("samples/*/")
hm_classes = len(sample_folders)

if __name__ == '__main__':

    with Pool(cpu_count()) as p:
        _ = p.map_async(class_pars_fn, enumerate(sample_folders))
        p.close()
        p.join()
        _.get()