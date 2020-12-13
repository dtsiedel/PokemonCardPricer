import cv2
import numpy as np

SCALER = 0.4
MAXWIDTH = round(700 * SCALER)
MAXHEIGHT = round(990 * SCALER)
BKG_THRESH = 80
CARD_MIN_AREA = 5000
CARD_MAX_AREA = 100000


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down SCALERx(MAXWIDTHxMAXHEIGHT) 
    perspective. Returns the flattened, re-sized"""
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


def process_cards(cnts_sort, cnt_is_card, frame, deck):
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
           
            # TODO: add back card match logic
            #matched_card_name = match_card(img, deck)
            #if matched_card_name is not None:
                #matched_card_img = deck.card_images[matched_card_name]
                #matched_card_img = cv2.resize(matched_card_img, (MAXWIDTH, MAXHEIGHT))
                #numpy_horizontal_concat = np.concatenate((img, matched_card_img), axis=1)
            #else:
                #numpy_horizontal_concat = np.concatenate((img, img*0), axis=1)
            #card_pairs.append(numpy_horizontal_concat)
        i += 1
        
    return card_pairs


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()     
        thresh = preprocess_img(frame)
        cnts_sort, cnt_is_card = find_cards(thresh)
        card_pairs = process_cards(cnts_sort, cnt_is_card, frame, deck=None)

        cv2.imshow('Card Search', frame)
        cv2.imshow('thresh', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
