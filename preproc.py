from glob import glob

from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

from scipy.io.wavfile import read
from scipy import signal

from pickle import dump



hm_frequencies = 3               # frequencies per timestep
hm_info        = 2               # hz & db info per frequency
info_normalize = [0.5,400000.0]  # max values per info



def wav_process(wav):

    try:
        rate, data = read(wav)
        hm_channels = data.shape[1]

        tracks = [data[:,i] for i in range(hm_channels)]
        tracks_converted = []

        for track in tracks:
            track_converted = []

            freqs, times, amps = signal.spectrogram(track)

            for time in range(len(times)):
                time_converted = []

                amps_at_time = amps[:,time]
                picked_freqs = amps_at_time.argsort()[-hm_frequencies:]

                for f in picked_freqs:
                    time_converted.append(freqs[f]/info_normalize[0])
                    time_converted.append(amps_at_time[f]/info_normalize[1])

                track_converted.append(time_converted)
            tracks_converted.append(track_converted)
        return tracks_converted

    except Exception as e:
        print(e)



if __name__ == '__main__':

    with ThreadPool(cpu_count()) as p:

        promise = p.map_async(wav_process, glob('data/*.wav'))
        p.close()
        p.join()

        dataset = []
        for tracks_converted in promise.get():
            if tracks_converted:
                dataset.extend(tracks_converted)

        dump(dataset, open('data.pkl','wb+'))
