import nltk
from flask import Flask, jsonify, url_for,request
import cv2
import pytesseract
from pip._vendor import requests
from skimage import io
from PIL import Image
import numpy as np
import mimetypes
import colorsys
from urllib.request import urlopen
from os.path import basename
import sys
import os
from subprocess import call
from io import StringIO
import scipy.misc
import time
import pandas as pd
from plotnine import *
from flask_cors import CORS
import json
from plotnine.data import *
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
import requests
from requests.exceptions import RequestException
from werkzeug.utils import secure_filename

import os
from utils import *
from darknet import Darknet
from scholarly import scholarly
app = Flask(__name__)
CORS(app)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('twitter_samples')
# nltk.download('stopwords')
# nltk.download('wordnet')


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# os.environ["TESSDATA_PREFIX"] = "C:\\Users\\User\\AppData\\Local\\Tesseract-OCR\\tessdata"
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\User\AppData\Local\Tesseract-OCR\tesseract.exe'
# the neural network configuration
config_path = "cfg/yolov3.cfg"
# the YOLO net weights file
weights_path = "weights/yolov3.weights"

# loading all the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
# Set the NMS Threshold
score_threshold = 0.6
# Set the IoU threshold
iou_threshold = 0.4
cfg_file = "cfg/yolov3.cfg"
weight_file = "weights/yolov3.weights"
namesfile = "data/coco.names"
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)


# baseUrl='http://vmi425296.contaboserver.net:5000/'
baseUrl='http://localhost:5000'
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)




@app.route('/')
def hello_world():
    return 'Hello World!'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
@app.route('/grayscale' , methods = ['GET', 'POST'])
def grascaleImage():
    # try:
        
    url='https://techcrunch.com/wp-content/uploads/2015/10/screen-shot-2015-10-08-at-4-20-16-pm.png?w=730&crop=1'
    # print (request)
    url=request.form['data']
    # f = request.files['imgFile']
    print (url)
    originalImage = io.imread(url)
    # page = requests.get(url)
    response = requests.get(url)
    content_type = response.headers['content-type']
    extension = mimetypes.guess_extension(content_type)
    if (extension == '.jpe'):
        extension = '.jpg'
    f_ext = os.path.splitext(url)[-1]
    # f_name = 'img{}'.format(f_ext)
    timeStr=str(int(round(time.time() * 1000)))
    f_name=timeStr+'gray'+extension
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, grayImage)

    return baseUrl+url_for('static',filename=f_name)

    # except:
    #     return jsonify(
    #         message='success',
    #         data='Put an image link with extension',
    #     )

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return int(h), int(s), int(v)
def rgb_hsv_converter(rgb):
    (r,g,b) = rgb_normalizer(rgb)
    hsv = colorsys.rgb_to_hsv(r,g,b)
    (h,s,v) = hsv_normalizer(hsv)
    upper_band = [h+10, s+40, v+40]
    lower_band = [h-10, s-40, v-40]
    return {
        'upper_band': upper_band,
        'lower_band': lower_band
    }

def rgb_normalizer(rgb):
    (r,g,b) = rgb
    return (r/255, g/255, b/255)

def hsv_normalizer(hsv):
    (h,s,v) = hsv
    return (h*360, s*255, v*255)

@app.route('/colorMasking', methods=['GET', 'POST'])
def colorMasking():
    try:

        red1 = request.form['red1']
        blue1 = request.form['blue1']
        green1 = request.form['green1']
        red2 = request.form['red2']
        blue2 = request.form['blue2']
        green2 = request.form['green2']
        uploaded_file = request.files['imgFile']
        timeStr = str(int(round(time.time() * 1000)))

        if uploaded_file.filename != '':
            filename = timeStr+secure_filename(filename = secure_filename(uploaded_file.filename))
            uploaded_file.save(os.path.join(UPLOAD_FOLDER, filename))
            img = cv2.imread('static/' + filename)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            # lowerRange = (red1, blue1, green1)
            # upperRange = (red2, blue2, green2)
            # plt.imshow(hsv_img)

            lowHsv=(rgb_to_hsv(float(red1), float(green1), float(blue1)))
            upHsv=(rgb_to_hsv(float(red2), float(green2), float(blue2)))
            print (lowHsv)
            print(upHsv)
            lowerRangeHSV= (int(float(lowHsv[0])/2), int(255*(float(lowHsv[1])/100)), int(255*(float(lowHsv[2])/100)))
            # upperRangeHSV=(float(red2)/2,float(green2), float(blue2))
            upperRangeHSV= (int(float(upHsv[0])/2), int(255*(float(upHsv[1])/100)), int(255*(float(upHsv[2])/100)))

            if(lowerRangeHSV>upperRangeHSV):
                temp=lowerRangeHSV
                lowerRangeHSV=upperRangeHSV
                upperRangeHSV=temp

            # print(lowerRangeHSV)
            # print(upperRangeHSV)
            # print ("new")
            # # lowerRangeHSV=(1, 190, 200)
            # # upperRangeHSV=(18, 255, 255)
            # print(lowerRangeHSV)
            # print(upperRangeHSV)

            mask = cv2.inRange(hsv_img, lowerRangeHSV, upperRangeHSV)
            plt.imshow(mask)
            plt.show()
            result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
            bgrResult=cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            filename = 'maksed'+filename
            cv2.imwrite(UPLOAD_FOLDER + '/' + filename, bgrResult)
            return app.send_static_file(filename)
        else:
            return jsonify(
                message='failure',
                data='No file found',
                status=500,
            )

    except Exception as e:
        return jsonify(
            message='failure',
            data='An error occoured',
            error=str(e),
            status=500,
        )


@app.route('/ocrAny', methods=['GET', 'POST'])
def ocrAny():
    try:
        language = request.form['language']
        uploaded_file = request.files['imgFile']
        timeStr = str(int(round(time.time() * 1000)))
        if uploaded_file.filename != '':
            filename = timeStr+secure_filename(filename = secure_filename(uploaded_file.filename))
            uploaded_file.save(os.path.join(UPLOAD_FOLDER, filename))
            img = cv2.imread('static/' + filename)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text=pytesseract.image_to_string(gray,lang=language)
            return text
        else:
            return jsonify(
                message='failure',
                data='No file found',
                status=500,
            )

    except Exception as e:
        return jsonify(
            message='failure',
            data='An error occoured',
            error=str(e),
            status=500,
        )

@app.route('/operationOnCSV', methods=['GET', 'POST'])
def operationOnCSV ():
    # try:
    operation = request.form['operation']
    isMultiple = request.form['isMultiple']
    uploaded_file = request.files['csvFile']
    timeStr = str(int(round(time.time() * 1000)))

    if uploaded_file.filename != '':
        filename = timeStr + secure_filename(filename=secure_filename(uploaded_file.filename))
        uploaded_file.save(os.path.join(UPLOAD_FOLDER, filename))
        df=pd.read_csv('static/' + filename)
        # print(df)
        numbers=df.iloc[:, 0]
        result=0
        if(len(numbers)>0):
            if (operation == 'min'):
                result=numbers[0]
            elif(operation=='mul'):
                result=1
        for number in numbers:
            print(number)
            if(operation=='add'):
                result=result+number
            elif(operation=='mul'):
                result=result*number
            elif(operation=='max'):
                if(result<number):
                    result=number
            else:
                if(number<result):
                    result=number

        if (isMultiple=='multiple'):
            numbersSecond = df.iloc[:, 1]
            for number in numbersSecond:
                print(number)
                if (operation == 'add'):
                    result = result + number
                elif (operation == 'mul'):
                    result = result * number
                elif (operation == 'max'):
                    if (result < number):
                        result = number
                else:
                    if (number < result):
                        result = number

        return str(result)
    else:
        return jsonify(
            message='failure',
            data='No file found',
            status=500,
        )

    # except Exception as e:
    #     return jsonify(
    #         message='failure',
    #         data='An error occoured',
    #         error=str(e),
    #         status=500,
    #     )

@app.route('/edgeDetectionFile', methods=['GET', 'POST'])
def edgeDetectionFile():
    try:
        uploaded_file = request.files['imgFile']
        timeStr = str(int(round(time.time() * 1000)))
        if uploaded_file.filename != '':
            filename = timeStr+secure_filename(filename = secure_filename(uploaded_file.filename))
            uploaded_file.save(os.path.join(UPLOAD_FOLDER, filename))
            img = cv2.imread('static/' + filename)
            edges = cv2.Canny(img, 100, 200)
            f_name = timeStr + 'edge' + uploaded_file.filename

            # calculate the edges using Canny edge algorithm
            # plot the edges


            cv2.imwrite(UPLOAD_FOLDER + '/' + f_name, edges)
            return app.send_static_file(f_name)
        else:
            return jsonify(
                message='failure',
                data='No file found',
                status=500,
            )

    except Exception as e:
        return jsonify(
            message='failure',
            data='An error occoured',
            error=str(e),
            status=500,
        )
@app.route('/edgeDetection' , methods = ['GET', 'POST'])
def edgeDetection():
    try:
        url='https://techcrunch.com/wp-content/uploads/2015/10/screen-shot-2015-10-08-at-4-20-16-pm.png?w=730&crop=1'
        url = request.form['image']
        originalImage = io.imread(url)
        page = requests.get(url)
        response = requests.get(url)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        if (extension == '.jpe'):
            extension = '.jpg'
        f_ext = os.path.splitext(url)[-1]
        f_name = 'img{}'.format(f_ext)
        timeStr=str(int(round(time.time() * 1000)))

        f_name=timeStr+'edge'+extension


        # calculate the edges using Canny edge algorithm
        # plot the edges

        edges = cv2.Canny(originalImage, 100, 200)

        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, edges)

        return baseUrl+url_for('static', filename=f_name)

    except:
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )

@app.route('/filterImage' , methods = ['GET', 'POST'])
def filterImage():
    try:
        

        url='https://docs.gimp.org/2.8/en/images/filters/examples/noise/taj-rgb-noise.jpg'
        url = request.form['imgdata']
        originalImage = io.imread(url)
        page = requests.get(url)
        timeStr=str(int(round(time.time() * 1000)))

        response = requests.get(url)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        if (extension == '.jpe'):
            extension = '.jpg'
        f_ext = os.path.splitext(url)[-1]
        print(f_ext)
        f_name = 'img{}'.format(f_ext)
        f_name=timeStr+'filter'+extension
        print (f_name)

        dst = cv2.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)

        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, dst)

        return baseUrl+url_for('static', filename=f_name)

    except:
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )


@app.route('/convertImageUsingUrl', methods=['GET', 'POST'])
def convertImageUsingUrl():
    try:

        url = request.form['imgdata']
        ext = request.form['ext']
        print(url)
        timeStr = str(int(round(time.time() * 1000)))

        response = requests.get(url)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        if (extension == '.jpe'):
            extension = '.jpg'
        f_ext = os.path.splitext(url)[-1]
        f_name = 'img{}'.format(f_ext)
        f_name = timeStr + 'oo.' + ext
        o_name = timeStr + 'oo' + extension
        print("assd")
        img_data = response.content
        with open(UPLOAD_FOLDER + '/' + o_name, 'wb') as handler:
            handler.write(img_data)
        # cv2.imwrite(UPLOAD_FOLDER + '/' + o_name, originalImage)
        im = Image.open(UPLOAD_FOLDER + '/' + o_name)
        im.save(UPLOAD_FOLDER + '/' + f_name)
        print ("assd")
        return baseUrl + url_for('static', filename=f_name)

    except Exception as e:
        print (e)
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )

@app.route('/objectDetection' , methods = ['GET', 'POST'])
def objectDetection():
    try:
        custom_tweet = request.form['data']
        eachUrl = custom_tweet.split(",")
        tableHtml='<table><thead><tr><th>Original image</th><th>Detected objects</th><th>Image with box plots</th><tbody>'
        for url in eachUrl:
            try:
                print(url.strip())
                originalImage = io.imread(url.strip())

                page = requests.get(url.strip())
                timeStr = str(int(round(time.time() * 1000)))

                f_ext = os.path.splitext(url)[-1]
                f_name = 'img{}'.format(f_ext)
                f_name = timeStr + 'filter' + f_name.split("?")[0]
                # f_name=timeStr+'filter'+f_ext
                response = requests.get(url)
                content_type = response.headers['content-type']
                extension = mimetypes.guess_extension(content_type)
                if(extension=='.jpe'):
                    extension='.jpg'
                fistName = 'img{}'.format(f_ext)
                fistName = timeStr + 'first' + fistName.split("?")[0]
                outputName = timeStr + 'out' + extension


                # cv2.imwrite(UPLOAD_FOLDER + '/' + fistName, originalImage)
                # original_image = cv2.imread('static/'+fistName)
                # original_image = cv2.cvtColor(originalImage, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(originalImage, (m.width, m.height))
                # detect the objects
                boxes = detect_objects(m, img, iou_threshold, score_threshold)
                # plot the image with the bounding boxes and corresponding object class labels
                detecteds= plot_boxes(originalImage, boxes, class_names,UPLOAD_FOLDER + '/' +outputName, plot_labels=True)
                tableHtml=tableHtml+'<tr><td><img src="'+url+'" style="max-height:100%; max-width:100%"><br>'+url+'</td>'+'<td>'+str(detecteds)+'</td>'+'<td><img src="'+baseUrl+url_for('static', filename=outputName)+'" style="max-height:100%; max-width:100%"><br>'+baseUrl+url_for('static', filename=outputName)+'</td></tr>'
            except Exception as e:
                print(e)

        tableHtml=tableHtml+'</tbody></table>'
        return tableHtml
        # url='https://docs.gimp.org/2.8/en/images/filters/examples/noise/taj-rgb-noise.jpg'
        # url = request.form['imgdata']
        # originalImage = io.imread(url)
        # page = requests.get(url)
        # timeStr=str(int(round(time.time() * 1000)))
        #
        # f_ext = os.path.splitext(url)[-1]
        # print(f_ext)
        # f_name = 'img{}'.format(f_ext)
        # f_name=timeStr+'filter'+f_name.split("?")[0]
        # print (f_name)
        #
        # dst = cv2.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)
        #
        # cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, dst)
        # return jsonify(
        #     message='success',
        #     data=baseUrl+url_for('static', filename=f_name),
        # )
    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )


@app.route('/masking' , methods = ['GET', 'POST'])
def masking():
    try:
        url='https://media.geeksforgeeks.org/wp-content/uploads/20190724220951/cat_damaged.png'
        #
        url2='https://media.geeksforgeeks.org/wp-content/uploads/20190724221007/cat_mask.png'

        # url='https://user-images.githubusercontent.com/25244255/42891003-af147bfc-8ac7-11e8-8440-29835a361782.png'
        # url2='https://user-images.githubusercontent.com/25244255/42891001-aea79712-8ac7-11e8-947f-cdfeae23de40.png'
        # url='http://www.vision.huji.ac.il/shiftmap/inpainting/jump.orig.jpg'
        # url2='http://www.vision.huji.ac.il/shiftmap/inpainting/jump.mask3.jpg'
        # url = request.form['imgdata']

        # url = 'http://localhost:5000/static/main.png'
        # url2 = 'http://localhost:5000/static/Untitled.png'
        originalImage = io.imread(url)
        url2originalImage = io.imread(url2)
        page = requests.get(url)
        timeStr=str(int(round(time.time() * 1000)))

        f_ext = os.path.splitext(url)[-1]
        print(f_ext)
        f_name = 'img{}'.format(f_ext)
        f_name=timeStr+'filter'+f_name.split("?")[0]
        # f_name=timeStr+'filter'+f_ext
        fistName = 'img{}'.format(f_ext)
        fistName = timeStr + 'first' + fistName.split("?")[0]
        cv2.imwrite(UPLOAD_FOLDER + '/' + fistName, originalImage)

        f_extnext = os.path.splitext(url)[-1]
        print(f_extnext)
        # url2originalImage=cv2.bitwise_not(url2originalImage)
        secondImage = 'img{}'.format(f_extnext)
        secondImage = timeStr + 'sec' + secondImage.split("?")[0]
        cv2.imwrite(UPLOAD_FOLDER + '/' + secondImage, url2originalImage)
        # # Open the image.
        img = cv2.imread('static/'+fistName)
        #
        # # Load the mask.
        mask = cv2.imread('static/'+secondImage, 0)

        # Inpaint.
        # im1=scipy.misc.toimage(originalImage)
        # im2=scipy.misc.toimage(url2originalImage)
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

        # # Write the output.
        # cv2.imwrite('cat_inpainted.png', dst)
        # dst = cv2.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)

        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, dst)
        return jsonify(
            message='success',
            # data=baseUrl+url_for('static', filename=f_name),
            data=baseUrl+url_for('static', filename=f_name),
        )
    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )


@app.route('/postagger' , methods = ['GET', 'POST'])
def postagger():
    try:

        inputType=request.args.get('inptype')
        outputType=request.args.get('outtype')
        eachtweet=[]
        custom_tweet = "I am having fun. , I was sad, I am going"


        if(inputType=='json'):
            content = request.json
            for j in content['inputStrings']:
                eachtweet.append(j)
        else:
            custom_tweet = request.form['data']
            eachtweet = custom_tweet.split(",")



        dictitnary={}

        for sentence in eachtweet:
            tokens = nltk.word_tokenize(sentence)
            tags=nltk.pos_tag(tokens)
            dictitnary[sentence]=tags
        return jsonify(
            message='success',
            data=dictitnary,
        )
    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='The input was invalid',
        )

@app.route('/sentimentAnalysis' , methods = ['GET', 'POST'])
def sentimentAnalysis():
    try:
        custom_tweet = "I am having fun. , I was sad, I am going"
        inputType = request.args.get('inptype')
        outputType = request.args.get('outtype')
        key = request.args.get('key')

        if(key=='12345'):
            eachtweet = []
            custom_tweet = "I am having fun. , I was sad, I am going"

            if (inputType == 'json'):
                content = request.json
                for j in content['inputStrings']:
                    eachtweet.append(j)
            else:
                custom_tweet = request.form['data']
                eachtweet = custom_tweet.split(",")

            dictitnary={}

            for sentence in eachtweet:
                custom_tokens = remove_noise(word_tokenize(sentence))
                response=classifier.classify(dict([token, True] for token in custom_tokens))
                dictitnary[sentence]=response

            print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
            return jsonify(
                message='success',
                data=dictitnary,
            )
        else:
            return jsonify(
                message='success',
                data='invalid_key',
            )
    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='The input was invalid',
        )


@app.route('/tableDataManipulation' , methods = ['GET', 'POST'])
def tableDataManipulation ():
    try:
        dict = request.form
        for key in dict:
            print('form key '+key + dict[key])
        data1=request.form['data1']
        # data2=request.form['data2']
        action1=request.form['action1']
        inpData1 = StringIO(""+str(data1)
            +"")

        # inpData2 = StringIO("" + str(data2)
        #                     + "")
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print (action1)
        table1 = pd.read_csv(inpData1, sep="\t")
        # df2 = pd.read_csv(inpData2, sep="\t")
        # df1.drop(df1.columns[[5,6,7,8,9]],axis=1,inplace=True)

        # df1['Series_reference'] = df1['Series_reference'].map(lambda name: name.lower()) #lower case
        # df1['Data_value'] = df1['Data_value'] *2 #perform opertation
        # df1['Series_reference']=df1['Series_reference'].str.startswith("H")
        eachCommand = action1.split("||")
        try:
            for command in eachCommand:
                print (command.strip())
                exec(command.strip())
        except:
            print("Error occured")

        # df1 = df1[['name', 'age', 'state']]//reorder
        # df1.drop(columns=['age', 'name'])
        # df1=df1.drop(columns=['Suppressed','Series_title_4','STATUS','UNITS','Magnitude','Subject','Group','Series_title_1'],axis=1,inplace=True)
        print (table1)
        result=table1.to_csv(sep='\t')
        # print (df2)
        return  result
        # return pd.concat([df1,df2], axis=1).to_csv(sep='\t')
    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='The input was invalid',
        )

@app.route('/deleteLine' , methods = ['GET', 'POST'])
def deleteLine ():
    try:
        text =request.form['text']
        line=request.form['line']

        front = text.split('\n')

        print(front)# Split just before the doomed line.
        textRet=''
        i=int(line)
        for idx, val in enumerate(front):
            if(idx+1!=i):
                textRet=textRet+val+'\n'
        response=textRet
        return response
    except Exception as e:
        response="Error occured please check your input"
        return response

@app.route('/fileConversion' , methods = [ 'POST'])
def fileConversion():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'there is no file in form!'
            file1 = request.files['file']
            target = request.form['target']
            timeStr = str(int(round(time.time() * 1000)))
            fileName=timeStr+ file1.filename
            path = UPLOAD_FOLDER+"/"+fileName
            file1.save(path)
            outputFile = timeStr + "out."+target
            outPath = UPLOAD_FOLDER+"/"+outputFile

            call('pandoc -s '+path+' -o '+outPath, shell=True)
            return baseUrl + url_for('static', filename=outputFile)
    except Exception as e:
        response="Error occured please check your input\n"
        return response+str(e)


@app.route('/fileConversionAsci' , methods = [ 'POST'])
def fileConversionAsci():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'there is no file in form!'
            file1 = request.files['file']
            timeStr = str(int(round(time.time() * 1000)))
            fileName=timeStr+ file1.filename
            path = UPLOAD_FOLDER+"/"+fileName
            file1.save(path)
            outputFile = timeStr + "out.txt"
            outPath = UPLOAD_FOLDER+"/"+outputFile

            call('pandoc -s '+path+' -t asciidoc -o '+outPath, shell=True)
            return baseUrl + url_for('static', filename=outputFile)
    except Exception as e:
        response="Error occured please check your input\n"
        return response+str(e)

@app.route('/convertImage', methods=['GET', 'POST'])
def convertImage():
    try:

        file1 = request.files['file']
        ext = request.form['ext']
        timeStr = str(int(round(time.time() * 1000)))
        fileName = timeStr + file1.filename
        path = UPLOAD_FOLDER + "/" + fileName
        file1.save(path)
        f_name = timeStr + 'oo.' + ext

        im = Image.open(path)
        im.save(UPLOAD_FOLDER + '/' + f_name)
        return app.send_static_file(f_name)

    except Exception as e:
        print(e)
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )


@app.route('/fileConversionFile' , methods = [ 'POST'])
def fileConversionFile():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'there is no file in form!'
            file1 = request.files['file']
            target = request.form['target']
            timeStr = str(int(round(time.time() * 1000)))
            fileName=timeStr+ file1.filename
            path = UPLOAD_FOLDER+"/"+fileName
            file1.save(path)
            outputFile = timeStr + "out."+target
            outPath = UPLOAD_FOLDER+"/"+outputFile
            print (outPath)
            time.sleep(2)
            call('pandoc -s '+path+' -o '+outPath, shell=True)
            return app.send_static_file(outputFile)
    except Exception as e:
        response="Error occured please check your input"
        return response+str(e)


@app.route('/fileConversionAsciFile' , methods = [ 'POST'])
def fileConversionAsciFile():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'there is no file in form!'
            file1 = request.files['file']
            timeStr = str(int(round(time.time() * 1000)))
            fileName=timeStr+ file1.filename
            path = UPLOAD_FOLDER+"/"+fileName
            file1.save(path)
            outputFile = timeStr + "out.txt"
            outPath = UPLOAD_FOLDER+"/"+outputFile
            time.sleep(2)
            call('pandoc -s '+path+' -t asciidoc -o '+outPath, shell=True)
            return app.send_static_file(outputFile)

    except Exception as e:
        response="Error occured please check your input"
        response=response+str(e)
        return response

@app.route('/replaceText' , methods = ['GET', 'POST'])
def replaceText ():
    try:

        text =request.form['text']
        word=request.form['word']
        replace=request.form['replaceWord']
        text=text.replace(word,replace)
        response=text
        return response
    except Exception as e:
        response="Error occured please check your input"
        return response

@app.route('/scholarlyBookAuthor' , methods = ['GET', 'POST'])
def scholarlyBookAuthor ():
    # try:
    query = request.form['query']
    search_query = scholarly.search_author(query)
    string = '['
    author = next(search_query).fill()
    for pub in author.publications:
        search_book = scholarly.search_pubs(pub.bib['title'])
        book=next(search_book)
        # print (book)
        url=''
        try:
            url=book['blb']
        except:
            url=''

        print (url)
        # print (basename(url))
    # for i in range(5):
    #     try:
    #         author = next(search_query)
    #         string += str(author) + ","
    #     except:
    #         print('')
    #
    # if (len(string) > 0):
    #     string = string[:-1]
    #     string=string+"]"

    return (str(string))
    # except Exception as e:
    #     response="Error occured please check your input"
    #     return response


@app.route('/scholarlyBookKeyword', methods = ['GET', 'POST'])
def scholarlyBookKeyword ():
    try:
        query = request.form['query']
        search_query = scholarly.search_keyword(query)
        string=''
        for i in range(5):
            try:
                author = next(search_query)
                string+=author.blb['eprint']
            except:
                print('')


        if(len(string)>0):
            string =string[:-1]

        return json.loads(str(string))
    except Exception as e:
        response="Error occured please check your input"
        return response

@app.route('/scholarlyBookPubs', methods = ['GET', 'POST'])
def scholarlyBookPubs ():
    try:
        query = request.form['query']
        search_query = scholarly.search_pubs(query)

        string=''
        for i in range(5):
            try:
                author = next(search_query)
                print (author['blb'])
                string=author.blb['eprint']
                print (string)
            except:
                string=''
            if(string!=''):
                break


        return (str(string))
    except Exception as e:
        print (e)
        response="Error occured please check your input"
        return response

@app.route('/graphPlot' , methods = ['GET', 'POST'])
def graphPlot ():
    # try:
    dict = request.form
    for key in dict:
        print('form key '+key + dict[key])
    data1=request.form['data1']
    # data2=request.form['data2']
    action1=request.form['action1']
    inpData1 = StringIO(""+str(data1)
        +"")

    global graph
    # inpData2 = StringIO("" + str(data2)
    #                     + "")
    # graph=geom_blank(mapping=None, data=None, stat='identity', position='identity',
    #    na_rm=False, inherit_aes=True, show_legend=None)
    global table1
    table1= pd.read_csv(inpData1, sep="\t")

    # df2 = pd.read_csv(inpData2, sep="\t")
    # df1.drop(df1.columns[[5,6,7,8,9]],axis=1,inplace=True)
    timeStr = str(int(round(time.time() * 1000)))
    # df1['Series_reference'] = df1['Series_reference'].map(lambda name: name.lower()) #lower case
    # df1['Data_value'] = df1['Data_value'] *2 #perform opertation
    # df1['Series_reference']=df1['Series_reference'].str.startswith("H")
    eachCommand = action1.split("||")
    try:
        for command in eachCommand:
            print (command.strip())
            exec(command.strip(),globals())
    except Exception as e:
        print (e)
        print("Error occured")
    # graph = (ggplot(table1, aes('Firstname','height(m)'))
    #          + geom_col(fill='green')
    #          + theme_light()
    #          )
    # df1 = df1[['name', 'age', 'state']]//reorder
    # df1.drop(columns=['age', 'name'])
    # df1=df1.drop(columns=['Suppressed','Series_title_4','STATUS','UNITS','Magnitude','Subject','Group','Series_title_1'],axis=1,inplace=True)
    # try:
    graph.save("static/"+timeStr+".png")
    return baseUrl + url_for('static', filename=timeStr+'.png')

    # except:
    #     return 'Error occured'
    # print (df2)

    # return pd.concat([df1,df2], axis=1).to_csv(sep='\t')
    # except Exception as e:
    #     print(e)
    #     return jsonify(
    #         message='success',
    #         data='The input was invalid',
    #     )

# if __name__ == "__main__":
#     app.run(host='0.0.0.0')
# if __name__ == '__main__':
#     app.run(debug=True)


