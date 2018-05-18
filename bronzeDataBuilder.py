import giphypop
import requests
from os import path, makedirs
from PIL import Image, ImagePalette
import io
import argparse
import json
import _pickle as pickle

limit = 100
giphy = giphypop.Giphy(api_key='NNfC3hdGO6xciq6jImRy17aLUjqzFIUf')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, help='Destination directory')
    return parser.parse_args()


# function takes a GIF and returns a list of images
def gif2jpg(gifbytes):
    try:
        img = Image.open(io.BytesIO(gifbytes))
    except IOError:
        return []

    i = 0
    palette = img.getpalette()
    images = []

    try:
        while True:
            img.putpalette(palette)
            next_image = Image.new("RGBA", img.size)
            next_image.paste(img)
            images.append(next_image.copy())
            i += 1
            img.seek(i)
    except EOFError:
        pass

    return images


# load and validate arguments
args = parse_args()
if not path.isdir(args.dest):
    makedirs(args.dest)
rootdir = args.dest
evaluated = {}

# load search words (type = list)
with open('words.json') as f:
    words = json.load(f)

start_index = 600

gifId = 14203

for index, term in enumerate(words[start_index:]):
    print('index %d: "%s"' % (index+start_index, term))
    num_new_gifs = 0
    for gif in giphy.search(term, limit=limit):
        try:
            url = gif.media_url

            if url not in evaluated:
                r = requests.get(url)
                data = r.content
                open(path.join(rootdir, "%i.gif" % gifId), "wb").write(data)
                evaluated[url] = gifId
                gifId += 1
                num_new_gifs += 1


        except Exception as e:
            print(e)

        print('  %d gifs added' % num_new_gifs)

with open('evaluated.pkl', 'wb') as f:
    pickle.dump(evaluated, f)

print('done')
