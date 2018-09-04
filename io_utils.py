import urllib
import json

def download_to_file(url, filename):
	f = urllib.URLopener()
	f.retrieve(url, filename)
	print "Downloading", filename

def load_as_json(filename):
	with open(filename, 'wb') as f:
		data = json.load(f)

	return data