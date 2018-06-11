import giphypop
import requests
from os import path, makedirs
from PIL import Image, ImagePalette
import io
import argparse
import json
import time

from Utils import init_dirs

limit = 100
giphy = giphypop.Giphy(api_key='')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to .JSON file with search terms')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory')
    parser.add_argument('--limit', type=int, default=100, help='Limit the number of GIFs for each search')
    parser.add_argument('--api-key', type=str, required=True, help='GIPHY API key')

    args = parser.parse_args()
    # verify limit value
    if args.limit <= 0:
        raise ValueError('parse_args: Invalid number for \'limit\'. Must be >= 1.')

    # unpack api
    if path.isfile(args.api_key):
        ext = path.splitext(args.api_key)[-1]
        if ext == '.json':
            j = json.load(args.api_key)
            if 'api' not in j:
                raise ValueError('parse_args: Cannot find API. JSON file does not contain key \'api\'.')
            args.api_key = j['api']
        elif ext == '.txt':
            with open(args.api_key) as f:
                args.api_key = f.readline().strip('\n').strip()
        else:
            raise ValueError('parse_args: Cannot find API. Unrecognized file extension')

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
dirnames = init_dirs()
logdir = dirnames['log']
resdir = dirnames['res']
configdir = dirnames['config']

if not path.isdir(args.dest):
    makedirs(args.dest)

rootdir = args.dest
setname = path.basename(path.splitext(args.dest)[0])
logfilename = path.join(logdir, '%s-Log.txt' % setname)
configfilename = path.join(configdir, '%s-Config.json' % setname)

# load search words (type = list)
with open(args.src) as f:
    categories = json.load(f)

if not isinstance(categories, dict):
    raise ValueError('Cannot infer .json format')

# load config file
if path.isfile(configfilename):
    with open(configfilename) as f:
        config = json.load(f)

    # mesh them together
    config_titles = list(config.keys())
    for config_tile in config_titles:
        for category_title in categories.keys():
            if category_title not in config:
                config[category_title] = 0
            else:
                idx = config_titles.index(category_title)
                del config_titles[idx]
    for title in config_titles:
        del config[title]

# create new config file
else:
    config = {
        title: {
            term: 0 for term in terms
        } for title, terms in categories.items()
    }
    with open(configfilename, 'w') as f:
        json.dump(config, f)
categories = config

if not path.isfile(logfilename):
    open(logfilename, 'w').close()


for category, terms in categories.items():

    # create new directory if needed
    outdir = path.join(rootdir, category)
    if not path.isdir(outdir):
        makedirs(outdir)

    # write log file
    logstr = 'category: %s\n' % category
    with open(logfilename, 'a+') as f:
        f.write(logstr)
    print(logstr[:-1])

    for term, start_index in terms.items():
        logstr = '  term       : %s\n' % term
        logstr += '  start index: %d\n' % start_index
        with open(logfilename, 'a+') as f:
            f.write(logstr)
        print(logstr[:-1])

        # query for each term
        start_t = time.time()
        success = 0
        num_url_failed = 0
        num_save_failed = 0
        for i, gif in enumerate(giphy.search(term, limit=limit)):
            if (i + 1) % 10 == 0:
                end_t = time.time()
                logstr = '    ------------------------\n'
                logstr += '    current index  : %d\n' % (i + start_index)
                logstr += '    gifs downloaded: %d\n' % success
                logstr += '    url fails      : %d\n' % num_url_failed
                logstr += '    save fails     : %d\n' % num_save_failed
                logstr += '    time           : %.2f\n' % (end_t - start_t)
                with open(logfilename, 'a+') as f:
                    f.write(logstr)
                print(logstr[:-1])

                success = 0
                num_url_failed = 0
                num_save_failed = 0

                # save config
                config[category][term] = i + start_index
                with open(configfilename, 'w') as f:
                    json.dump(config, f, indent=2)
                start_t = time.time()

            try:
                url = gif.media_url

                try:
                    r = requests.get(url)
                    data = r.content
                    filename = '%s_%d.gif' % (term, i)

                    open(path.join(outdir, filename), "wb+").write(data)

                    success += 1

                except Exception as e:
                    num_save_failed += 1
            except Exception as e:
                num_url_failed += 1


print('done')
