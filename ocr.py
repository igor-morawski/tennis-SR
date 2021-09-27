import cv2
import pytesseract
import re

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def median_blur(image):
    return cv2.medianBlur(image,5)
 
def otsu(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def opening(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

class OCR_Tesseract:
    def __init__(self, tesseract_config = "--oem 3 --psm 6 -c tessedit_char_whitelist=>abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", preprocessing_pipeline : list = None):
        self.tesseract_config = tesseract_config
        self.preprocessing_pipeline = preprocessing_pipeline
    
    def extract(self, image):
        if self.preprocessing_pipeline:
            for func in self.preprocessing_pipeline:
                image = func(image)
        return pytesseract.image_to_string(image, config=self.tesseract_config)

class StringProcessor():
    def __init__(self):
        self.alphabetic_replacements = [
            ("1", "l"),
            ("0", "o"),
            ("8", "b"),
        ]
        self.numerical_replacements = [
            ("o", "0"),
            ("O", "0"),
            ("l", "1"),
            ("B", "8"),
            ("?", "5")
        ]
    
    def preprocess_alphabetic(self, alpha_string):
        res = alpha_string
        for (c_0, c_r) in self.alphabetic_replacements:
            res = res.replace(c_0, c_r)
        return ''.join([i for i in res if i.isalpha()])
        
    def preprocess_numerical(self, num_string):
        res = num_string
        for (c_0, c_r) in self.numerical_replacements:
            res = res.replace(c_0, c_r)
        return ''.join([i for i in res if i.isdigit()])
        
    def read_values(self, scoreboard_string):
        return [self._read_values(s) for s in scoreboard_string.splitlines()[:2]]
    
    def _read_values(self, noisy_string):
        noisy_string = noisy_string
        player_name = []
        player_surname = []
        score = []
        s_split = re.split('\s|\.', noisy_string)
        # add name, surname, score
        surname_info = max(enumerate(s_split), key=lambda x: len(x[1]))
        surname_id = surname_info[0]
        player_surname.append(surname_info[1])
        for i, s in enumerate(s_split):
            if i < surname_id and s != '>':
                player_name.append(s)  # allow for several initials
            if i > surname_id:
                score.append(s)
        #clean up
        cleaned_name = [self.preprocess_alphabetic(
            x) for x in player_name if x != ""]
        cleaned_surname = [self.preprocess_alphabetic(
            x) for x in player_surname if x != ""]
        cleaned_score = [self.preprocess_numerical(x) for x in score if x != ""] 
        if '>' in s_split[0]:
            serving = True
        else:
            serving = False
        if cleaned_name:
            cleaned_name[0]+=". "
            initial = cleaned_name[0]
        else:
            initial = ""
        cleaned_score = "-".join(cleaned_score)
        if len(cleaned_surname):
            cleaned_surname = cleaned_surname[0]
        else: 
            cleaned_surname = ""
        return initial+cleaned_surname, cleaned_score, serving





