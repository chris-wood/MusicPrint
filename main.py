from pydub import AudioSegment
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
import numpy as np

# Sampling rate, related to the Nyquist conditions, which affects the range frequencies we can detect.
DEFAULT_FS = 44100

# Other defaults...
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20

def generateSpectogram(channel_samples, Fs=DEFAULT_FS, wsize=DEFAULT_WINDOW_SIZE, wratio=DEFAULT_OVERLAP_RATIO, fan_value=DEFAULT_FAN_VALUE, amp_min=DEFAULT_AMP_MIN):
	# a tuple (Pxx, freqs, t) is said to be the return value of specgram
	arr2D = mlab.specgram(channel_samples, NFFT=wsize, Fs=Fs, window=mlab.window_hanning, noverlap=int(wsize * wratio))[0]

	# apply log transform since specgram() returns linear array
	arr2D = 10 * np.log10(arr2D)
	arr2D[arr2D == -np.inf] = 0 # replace infs with zeros

	# find local maxima
	plotPeaks(arr2D, amp_min=amp_min)

def plotPeaks(arr2D, amp_min=DEFAULT_AMP_MIN):
	# http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
	struct = generate_binary_structure(2, 1)
	neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

	# find local maxima using our fliter shape
	local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
	background = (arr2D == 0)
	eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

	# Boolean mask of arr2D with True at peaks
	detected_peaks = local_max - eroded_background

	# extract peaks
	amps = arr2D[detected_peaks]
	j, i = np.where(detected_peaks)

	# filter peaks
	amps = amps.flatten()
	peaks = zip(i, j, amps)
	peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

	# get indices for frequency and time
	frequency_idx = [x[1] for x in peaks_filtered]
	time_idx = [x[0] for x in peaks_filtered]

	# scatter of the peaks
	fig, ax = plt.subplots()
	ax.imshow(arr2D)
	ax.scatter(time_idx, frequency_idx)
	ax.set_xlabel('Time')
	ax.set_ylabel('Frequency')
	ax.set_title("Spectrogram")
	plt.gca().invert_yaxis()
	plt.show()

def main(argv):
	songData = AudioSegment.from_mp3(argv[1])

	# Data is an array of values for each channel
	# e.g., 2 channels
	# val1chan1, val1chan2, val2chan1, val2chan2, ...
	data = np.fromstring(songData._data, np.int16)

	# Extract the sound data for each channel
	channels = []
	for chn in xrange(songData.channels):
		channels.append(data[chn::songData.channels])
	
	frameRate = songData.frame_rate
	for c in channels:
		generateSpectogram(c, Fs = frameRate)

if __name__ == "__main__":
	if (len(sys.argv) == 2):
		main(sys.argv)
	else:
		print >> sys.stderr, "Usage: main.py path-to-mp3-file"
