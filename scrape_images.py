import requests
from bs4 import BeautifulSoup as soup

import os
import pathlib
import re
import sys
import shutil
from typing import Callable


class PkmnSet:
    def __init__(self, desc: str, input_regex: re.Pattern, transform: Callable=None):
        # Name of the set, format to match the "input file name" (match entire string), transformation to get "output rel url for limitlesstcg.com"
        # Our images locally mirror the layout of limitlesstcg.com
        self.desc = desc
        self.input_regex = input_regex
        self.transform = PkmnSet.standard_transform if transform is None else transform

    def matches(self, string) -> bool:
        return bool(re.fullmatch(self.input_regex, string))

    def get_url(self, string) -> str:
        # caller responsible for checking that the string matches the format
        return self.transform(string)

    @staticmethod
    def standard_transform(string) -> str:
        parts = string.split('-')
        return f'/{parts[-2]}/{parts[-1][:-1]}'


sets = [
    PkmnSet('Vivid Voltage', re.compile('.*-viv-\d+a?b?/')),
    PkmnSet('Chamption\'s Path', re.compile('.*-cpa-\d+a?b?/')),
    PkmnSet('Darkness Ablaze', re.compile('.*-daa-\d+a?b?/')),
    PkmnSet('Rebel Clash', re.compile('.*-rcl-\d+a?b?/'), lambda s: str(int(f'/{parts[-2]}/{parts[-1][:-1]}'))),
    PkmnSet('Sword & Shield', re.compile('.*-ssh-\d+a?b?/')),
    PkmnSet('Cosmic Eclipse', re.compile('.*-cec-\d+a?b?/')),
    PkmnSet('Hidden Fates', re.compile('.*-hif-\d+a?b?/')),
    PkmnSet('Unified Minds', re.compile('.*-unm-\d+a?b?/')),
    PkmnSet('Unbroken Bonds', re.compile('.*-unb-\d+a?b?/')),
    PkmnSet('Detective Pikachu', re.compile('.*-det-\d+a?b?/')),
    PkmnSet('Team Up', re.compile('.*-teu-\d+a?b?/')), 
    PkmnSet('Lost Thunder', re.compile('.*-lot-\d+a?b?/')),
    PkmnSet('Dragon Majesty', re.compile('.*-drm-\d+a?b?/')),
    PkmnSet('Celestial Storm', re.compile('.*-ces-\d+a?b?/')),
    PkmnSet('Forbidden Light', re.compile('.*-fli-\d+a?b?/')),
    PkmnSet('Ultra Prism', re.compile('.*-upr-\d+a?b?/')),
    PkmnSet('Crimson Invasion', re.compile('.*-cin-\d+a?b?/')),
    PkmnSet('Shining Legends', re.compile('.*-slg-\d+a?b?/')),
    PkmnSet('Burning Shadows', re.compile('.*-bus-\d+a?b?/')),
    PkmnSet('Guardians Rising', re.compile('.*-gri-\d+a?b?/')),
    PkmnSet('Sun & Moon', re.compile('.*-sum-\d+a?b?/')),
    # TODO: standard format for shiny vaults
    PkmnSet('Team Up Shiny Vault', re.compile('.*-teu-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),  # shiny vault can be weird
    PkmnSet('Dragon Majesty Shiny Vault', re.compile('.*-drm-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Celestial Storm Shiny Vault', re.compile('.*-ces-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Forbidden Light Shiny Vault', re.compile('.*-fli-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Lost Thunder Shiny Vault', re.compile('.*-lot-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Ultra Prism Shiny Vault', re.compile('.*-upr-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Sun & Moon Shiny Vault', re.compile('.*-sum-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Burning Shadows Shiny Vault', re.compile('.*-bus-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Guardians Rising Shiny Vault', re.compile('.*-gri-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Shining Legends Shiny Vault', re.compile('.*-slg-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Crimson Invasion Shiny Vault', re.compile('.*-cin-sv\d+a?b?/'), lambda s: f'/hif/{s.split("-")[-1][:-1]}'),
    PkmnSet('Sun & Moon Promos', re.compile('.*-sun-moon-promos-sm\d+a?b?/'), lambda s: f'/smp/{s.split("-")[-1][:-1]}'),
    PkmnSet('Sword & Shield Promos', re.compile('.*-sword-shield-promos-swsh\d+/'), lambda s: f'/ssp/{str(int(s.split("-")[-1][4:-1]))}'),
]
skip_sets = [
    PkmnSet('Sun & Moon Trainer Kit Raichu', re.compile('.*-tk10a-\d+a?b?/')),
    PkmnSet('Sun & Moon Trainer Kit Lycanroc', re.compile('.*-tk10l-\d+a?b?/')),
    PkmnSet('All Energies', re.compile('.*-energy/')),
]
        

def get_filename(url: str) -> str:
    for s in sets:
        if s.matches(url):
            return s.get_url(url)
    for s in skip_sets:
        if s.matches(url):
            return None
    raise Exception('No set for', url)


image_dir = 'images'
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
page_urls = []
i = 1
while True:
    url = f'https://pkmncards.com/page/{i}/?s=format%3Ateu-on-standard-2021%2Cupr-on-standard-2020%2Csum-on-standard-2019&display=images&sort=date&order=asc'
    # TODO: some shiny vault misbehaving. Find a way to work it into main URL
    #url = f'https://pkmncards.com/page/{i}/?s=collection%3Ashiny-vault&display=images&sort=date&order=asc'

    r = requests.get(url)
    if r.status_code != 200:
        print('assuming we stopped with', i - 1, 'pages')
        break
    page = soup(r.content, 'html.parser')

    entries = page.find_all('div', {'class': 'entry-content'})
    page_urls.extend([e.a['href'] for e in entries])

    print(f'Fetched page', i)
    i += 1


for url in page_urls:
    r = requests.get(url)
    r.raise_for_status()
    page_soup = soup(r.content, 'html.parser')
    card_img_url = page_soup.find("div", {"class": "card-image"}).a['href']

    r = requests.get(card_img_url, stream=True)
    r.raise_for_status()

    local_rel = get_filename(url)
    if local_rel is None:  # indicates a skip set
        print('skipping marked url', url)
        continue
    local_filename = f'{image_dir}{local_rel}'
    if os.path.exists(local_filename):
        print('skipping already-saved', local_filename)
        continue
    parent = str(pathlib.Path(local_filename).parent)
    if not os.path.exists(parent):
        os.mkdir(parent)

    with open(local_filename, 'wb') as file_out:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, file_out)

    print('Wrote image', file_out)


print('Done.')
