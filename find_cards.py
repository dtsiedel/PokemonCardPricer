import argparse
import collections
import functools
import glob
import os
import pathlib
import threading

import bs4
import cv2
import numpy as np
import requests


SCALER = 0.4
MAXWIDTH = round(700 * SCALER)
MAXHEIGHT = round(990 * SCALER)
BKG_THRESH = 80
CARD_MIN_AREA = 5000
CARD_MAX_AREA = 100000
IMG_ROOT = str(pathlib.Path(os.getcwd()) / 'images')
SIFT_OBJ = cv2.SIFT_create()
SHORTCUT_MATCH_THRESH = 10
LOOKUP_URL = 'https://limitlesstcg.com/cards/'


class Card:
    """Note that 'path' doubles as the path to the local file and the relative
    path to the card on limitlesstcg.com"""
    def __init__(self, path, img_data):
        self.path = path
        self.img_data = img_data
        self.keypoints = None

    def __repr__(self):
        return str(self.__dict__)

    def load_keypoints(self, sift):
        gray = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15,15), 0)
        kp, des = sift.detectAndCompute(blur, None)
        self.keypoints = (kp, des)


def get_cards_in_deck(root, region_codes):
    """Given the root dir of the images tree, get a list of all of the
    files that contain cards."""
    # TODO: speed up matching in some way to allow for all regions at once
    # TODO: it should be possible to multithread some matching with Pool
    root_parts = pathlib.Path(root).parts
    all_files = []
    for code in region_codes:
        all_files.extend(glob.glob(f'{root}/{code}/**/*', recursive=True))
    card_paths = [f for f in all_files if not os.path.isdir(f)]
    
    # path as key replicates data but is a good optimization for lookups
    cards = {}
    for n in card_paths:
        rel = str(pathlib.Path(n).relative_to(*root_parts))
        this_card = Card(rel, cv2.imread(n))
        this_card.load_keypoints(SIFT_OBJ)
        cards[rel] = this_card
    print('Read', len(cards), 'images.')

    return cards

 
def load_card_keypoints(self, images):
    card_keypoints = []
    
    for card in cards:
        gray = cv2.cvtColor(card.img_data, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15,15), 0)
        kp, des = sift.detectAndCompute(blur, None)
        card_keypoints.append([kp, des, card.path])
    
    return card_keypoints


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down SCALERx(MAXWIDTHxMAXHEIGHT) 
    perspective. Returns the flattened, re-sized image"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8 * h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8 * h and w < 1.2 * h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [MAXWIDTH - 1, 0], [MAXWIDTH - 1, MAXHEIGHT - 1], [0, MAXHEIGHT - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (MAXWIDTH, MAXHEIGHT))

    return warp


def preprocess_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    bkg_level = gray[100, 100]
    thresh_level = bkg_level + BKG_THRESH
    retval, thresh_image = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
    
    return thresh_image


def find_cards(thresh):
    cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]), reverse=True)
    
    if len(cnts) == 0:
        return [], []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01*peri, True)

        if size < CARD_MAX_AREA and size > CARD_MIN_AREA\
            and hier_sort[i][3] == -1 and len(approx) == 4:
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def match_card(card_img, cards, expected=None):
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    
    bf = cv2.BFMatcher()
    max_good_points = 0
    max_pokemon_match = None

    if expected is not None:
        des2 = cards.get(expected).keypoints[1]
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        if len(good) > SHORTCUT_MATCH_THRESH:
            return expected

    for path, card in cards.items():
        pokemon = card.keypoints
        des2 = pokemon[1]
        
        matches = bf.knnMatch(des1, des2, k=2)
    
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        good_points = len(good)
        if good_points > max_good_points:
            max_good_points = good_points
            max_pokemon_match = path

    return max_pokemon_match


def process_cards(cnts_sort, cnt_is_card, frame, cards: dict, expected=None):
    i = 0
    card_pairs = []
    for is_card in cnt_is_card:
        if is_card:
            contour = cnts_sort[i]
            # Find perimeter of card and use it to approximate corner points
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
            pts = np.float32(approx)

            # Find width and height of card's bounding rectangle
            x,y,w,h = cv2.boundingRect(contour)

            # Warp card into SCALERx(900x770) flattened image using perspective transform
            img = flattener(frame, pts, w, h)

            cv2.imshow('cutout', img)
           
            matched_card_name = match_card(img, cards, expected=expected)
            if matched_card_name is not None:
                return matched_card_name
            else:
                return None
        i += 1


def threadsafe_lru(func):
    func = functools.lru_cache()(func)
    lock_dict = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        key = functools._make_key(args, kwargs, typed=False)
        with lock_dict[key]:
            return func(*args, **kwargs)

    return _thread_lru


@threadsafe_lru
def card_details_from_path(path):
    print('lookup', path)
    url = LOOKUP_URL + path
    r = requests.get(url)
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.content, 'html.parser')

    price = soup.find('span', class_="card-buy-button-price usd").text
    name = soup.find('span', class_="card-text-name").text

    return {'price': price, 'name': name}


def set_last_price(path, price_obj):
    details = card_details_from_path(path)
    price_obj['price'] = details['price']
    price_obj['name'] = details['name']


def annotate_frame(frame, last_price):
    if len(last_price) == 0:
        return

    string = f'{last_price["name"]} (${last_price["price"]})'
    cv2.putText(frame, string, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify Pokemon Cards.')
    parser.add_argument('-s', '--sets', nargs='+', help='<Required> Region Code', required=True)
    parser.add_argument('-c', '--consecutive-matches', type=int, default=3)
    args = parser.parse_args()

    cards = get_cards_in_deck(IMG_ROOT, args.sets)

    cap = cv2.VideoCapture(0)

    match_count = args.consecutive_matches
    last_matches = collections.deque([None] * match_count, match_count)
    last_price = {}  # use dict to allow mutation in thread
    while True:
        ret, frame = cap.read()     
        thresh = preprocess_img(frame)
        cnts_sort, cnt_is_card = find_cards(thresh)
        match = process_cards(cnts_sort, cnt_is_card, frame, cards, expected=last_matches[0])

        # True match, fetch price
        if match is not None and all([x == match for x in list(last_matches)]):
            th = threading.Thread(target=set_last_price, args=(match, last_price))
            th.start()

        # Remove price if no card
        if match is None and all([x == match for x in list(last_matches)]):
            last_price = {}
        last_matches.append(match)

        # TODO: commandline debug flag to show threshold, prints
        annotate_frame(frame, last_price)
        cv2.imshow('Card Search', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
