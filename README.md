# keyword-ocr
There are two *nearly* identical versions in this. One is a standalone that runs in console and the other is
a quick Discord chatbot.

### Discord
Enter your bot token in `config.py` and run

**Command syntax:** `$ocr <image url>`

### Standalone
Edit `url` in `main()` with desired URL **or** make that variable false and direct it to a locally stored 
image. This is just for testing but I might clean it up later.

### Status
✔: Accept Discord input via image URL

✔: Use an OCR to convert text in image to a string

❌: Filter out unnecessary words, leaving potential keywords

✔: Add image and keywords to a table for storage (Just going to reuse code from 
[here](https://github.com/ottter/dodo/blob/master/cogs/people.py).)

❌: Ability to retrieve image later through keywords (RNG and/or most relevant based on inputs)

### Resources
- https://realpython.com/natural-language-processing-spacy-python/
- https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
- https://nanonets.com/blog/ocr-with-tesseract/