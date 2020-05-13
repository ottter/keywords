import re
import cv2
import spacy
import pytesseract
import numpy as np
from pytesseract import Output

# ... or import a folder of example-images
# for image in os.listdir('example-images'):
#     img = cv2.imread(f"example-images/{image}")
#     text = pytesseract.image_to_string(img)
#     print(text.replace('\n', ' '))

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
        orig_text_pool.append(text)
        filtered_pool.append(filter_noise.sub('', text))
    processed_index = filtered_pool.index(max(filtered_pool, key=len))
    # Return the original text from the processed image with best results
    return orig_text_pool[processed_index]

def process_image(img):
    invert = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5, 5), 0)
    return [img, invert, gaus]

# Find the most dominant color in an image. Potentially useful because tesseract works better with light background
def dominant_color(img):
    img_2d = img.reshape(-1, img.shape[-1])
    col_range = (256, 256, 256)
    img_1d = np.ravel_multi_index(img_2d.T, col_range)
    return np.unravel_index(np.bincount(img_1d).argmax(), col_range)

def main():
    # Desired Keyword
    search_term = 'lamp-light'

    # Import image
    img_orig = cv2.imread("example-images/2.png")
    img_copy = img_orig.copy()

    processed_image = process_image(img_orig)

    processed_text = best_match(processed_image)
    print(processed_text)

    bounding_box(search_term, img_copy)

    # Word Filter
    # candidate_pos = ['NOUN', 'PROPN', 'VERB']
    # useful_words = []
    # doc = nlp(gaus_text)
    # for sent in doc.sents:
    #     selected_words = []
    #     for token in sent:
    #         if token.pos_ in candidate_pos and token.is_stop is False:
    #             selected_words.append(token)
    #     useful_words.append(selected_words)
    # print(useful_words)

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')

    # https://github.com/UB-Mannheim/tesseract/wiki
    # Install Tesseract from link above and replace path with your path to tesseract.exe (or add to PATH)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    main()

# https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
# https://nanonets.com/blog/ocr-with-tesseract/
# https://www.youtube.com/watch?v=v9X3j-2p4yA