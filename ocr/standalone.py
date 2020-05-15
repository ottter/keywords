import re
import cv2
import spacy
import requests
import pytesseract
import numpy as np
from pytesseract import Output

def keyword_list(processed_text):
    # Word Filter
    candidate_pos = ['NOUN', 'PROPN', 'VERB']
    useful_words, sent_list = [], []
    doc = nlp(processed_text.replace('\n', ' '))
    for sent in doc.sents:  # Sentence detection
        selected_words = []
        sent_list.append(str(sent))
        for token in sent:
            if token.pos_ in candidate_pos and not token.is_stop:
                selected_words.append(token.lemma_.strip().lower())
                # Lemmatization - reduce inflected forms of a word but keep meaning (keeps > keep, organizing > organize)
        useful_words.append(selected_words)
    keyword_list = [j for i in useful_words for j in i]
    return keyword_list

# Temporary, won't be used in final program
def bounding_box(search_term, img_copy):
    data = pytesseract.image_to_data(img_copy, output_type=Output.DICT)
    word_list = [i for i, word in enumerate(data["text"]) if word.lower() == search_term]
    for i in word_list:
        # extract the positions for that specified word
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        # define all the surrounding box coordinates
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        # draw the bounding box
        img_copy = cv2.line(img_copy, p1, p2, color=(255, 0, 255), thickness=1)
        img_copy = cv2.line(img_copy, p2, p3, color=(255, 0, 255), thickness=1)
        img_copy = cv2.line(img_copy, p3, p4, color=(255, 0, 255), thickness=1)
        img_copy = cv2.line(img_copy, p4, p1, color=(255, 0, 255), thickness=1)

    cv2.imshow('highlight_keyword', img_copy)
    cv2.waitKey(0)

def best_match(processed_image):
    orig_text_pool, filtered_pool = [], []
    for img in processed_image:
        text = pytesseract.image_to_string(img).replace('\n\n', '\n').replace('  ', ' ')
        filter_noise = re.compile(r'\W*\b\w{1,3}\b')
        # Filter out 1-3 character words to help find best match. NOTE: They are only temporarily removed
        orig_text_pool.append(text)
        filtered_pool.append(filter_noise.sub('', text))
    processed_index = filtered_pool.index(max(filtered_pool, key=len))
    # Return the original text from the processed image with best results
    return orig_text_pool[processed_index]

def process_image(img):
    invert = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5, 5), 0)
    return [img, invert, gaus]  # Pass multiple processed images through to later pick the best one

# Find the most dominant color in an image. Potentially useful because pytesseract works better with light background
def dominant_color(img):
    img_2d = img.reshape(-1, img.shape[-1])
    col_range = (256, 256, 256)
    img_1d = np.ravel_multi_index(img_2d.T, col_range)
    return np.unravel_index(np.bincount(img_1d).argmax(), col_range)

def main():
    # Desired Keyword
    search_term = 'lamp-light'

    # Import image
    url = r'https://i.imgur.com/qspVN6I.png'
    # url = False
    if url:
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_orig = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        img_orig = cv2.imread("example-images/5.png")
    img_copy = img_orig.copy()

    processed_image = process_image(img_orig)

    processed_text = best_match(processed_image)
    keywords = keyword_list(processed_text)
    print(keywords)
    bounding_box(search_term, img_copy)



if __name__ == "__main__":
    # https://github.com/UB-Mannheim/tesseract/wiki
    # Install Tesseract from link above and replace path with your path to tesseract.exe (or add to PATH)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    nlp = spacy.load('en_core_web_sm')

    main()