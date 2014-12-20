from pydub import AudioSegment
import sys

song = AudioSegment.from_mp3(sys.argv[1])