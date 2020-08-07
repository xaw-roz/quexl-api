import nltk
from flask import Flask, jsonify, url_for,request
import cv2
from pip._vendor import requests
from skimage import io
import numpy as np
import mimetypes
import sys
from io import StringIO
import scipy.misc
import time
import pandas as pd
from plotnine import *
from plotnine.data import *
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
import os
from utils import *
from darknet import Darknet

app = Flask(__name__)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('wordnet')


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


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


baseUrl='http://vmi425296.contaboserver.net:5000/'
# baseUrl='http://localhost:5000'
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
    try:
        
        url='https://techcrunch.com/wp-content/uploads/2015/10/screen-shot-2015-10-08-at-4-20-16-pm.png?w=730&crop=1'
        url=request.form['data']
        originalImage = io.imread(url)
        page = requests.get(url)
        response = requests.get(url)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        f_ext = os.path.splitext(url)[-1]
        f_name = 'img{}'.format(f_ext)
        timeStr=str(int(round(time.time() * 1000)))
        f_name=timeStr+'gray'+extension
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, grayImage)
        return jsonify(
            message='success',
            data=baseUrl+url_for('static',filename=f_name),
        )
    except:
        return jsonify(
            message='success',
            data='Put an image link with extension',
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
        f_ext = os.path.splitext(url)[-1]
        f_name = 'img{}'.format(f_ext)
        timeStr=str(int(round(time.time() * 1000)))

        f_name=timeStr+'edge'+extension


        # calculate the edges using Canny edge algorithm
        # plot the edges

        edges = cv2.Canny(originalImage, 100, 200)

        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, edges)
        return jsonify(
            message='success',
            data=baseUrl+url_for('static', filename=f_name),
        )
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
        f_ext = os.path.splitext(url)[-1]
        print(f_ext)
        f_name = 'img{}'.format(f_ext)
        f_name=timeStr+'filter'+extension
        print (f_name)

        dst = cv2.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)

        cv2.imwrite(UPLOAD_FOLDER+'/'+f_name, dst)
        return jsonify(
            message='success',
            data=baseUrl+url_for('static', filename=f_name),
        )
    except:
        return jsonify(
            message='success',
            data='Put an image link with extension',
        )@app.route('/filterImage' , methods = ['GET', 'POST'])

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
@app.route('/objectDetectionOpenCV' , methods = ['GET', 'POST'])
def objectDetectionOpenCV():
    try:
        # colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        #
        # # load the YOLO network
        # net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        #
        # # path_name = "images/city_scene.jpg"
        # path_name = sys.argv[1]
        # image = cv2.imread(path_name)
        # file_name = os.path.basename(path_name)
        # filename, ext = file_name.split(".")
        #
        # h, w = image.shape[:2]
        # # create 4D blob
        # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        #
        # # sets the blob as the input of the network
        # net.setInput(blob)
        #
        # # get all the layer names
        # ln = net.getLayerNames()
        # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # # feed forward (inference) and get the network output
        # # measure how much it took in seconds
        # start = time.perf_counter()
        # layer_outputs = net.forward(ln)
        # time_took = time.perf_counter() - start
        # print(f"Time took: {time_took:.2f}s")
        #
        # boxes, confidences, class_ids = [], [], []
        #
        # # loop over each of the layer outputs
        # for output in layer_outputs:
        #     # loop over each of the object detections
        #     for detection in output:
        #         # extract the class id (label) and confidence (as a probability) of
        #         # the current object detection
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         # discard weak predictions by ensuring the detected
        #         # probability is greater than the minimum probability
        #         if confidence > CONFIDENCE:
        #             # scale the bounding box coordinates back relative to the
        #             # size of the image, keeping in mind that YOLO actually
        #             # returns the center (x, y)-coordinates of the bounding
        #             # box followed by the boxes' width and height
        #             box = detection[:4] * np.array([w, h, w, h])
        #             (centerX, centerY, width, height) = box.astype("int")
        #
        #             # use the center (x, y)-coordinates to derive the top and
        #             # and left corner of the bounding box
        #             x = int(centerX - (width / 2))
        #             y = int(centerY - (height / 2))
        #
        #             # update our list of bounding box coordinates, confidences,
        #             # and class IDs
        #             boxes.append([x, y, int(width), int(height)])
        #             confidences.append(float(confidence))
        #             class_ids.append(class_id)
        #
        # # perform the non maximum suppression given the scores defined before
        # idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
        #
        # font_scale = 1
        # thickness = 1
        #
        # # ensure at least one detection exists
        # if len(idxs) > 0:
        #     # loop over the indexes we are keeping
        #     for i in idxs.flatten():
        #         # extract the bounding box coordinates
        #         x, y = boxes[i][0], boxes[i][1]
        #         w, h = boxes[i][2], boxes[i][3]
        #         # draw a bounding box rectangle and label on the image
        #         color = [int(c) for c in colors[class_ids[i]]]
        #         cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        #         text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        #         # calculate text width & height to draw the transparent boxes as background of the text
        #         (text_width, text_height) = \
        #         cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        #         text_offset_x = x
        #         text_offset_y = y - 5
        #         box_coords = (
        #         (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        #         overlay = image.copy()
        #         cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        #         # add opacity (transparency to the box)
        #         image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        #         # now put the text (label: confidence %)
        #         cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #                     fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        #
        # # cv2.imshow("image", image)
        # # if cv2.waitKey(0) == ord("q"):
        # #     pass
        #
        # cv2.imwrite(filename + "_yolo3." + ext, image)
        print ('s')
    except:
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

    # inpData2 = StringIO("" + str(data2)
    #                     + "")
    graph=geom_blank(mapping=None, data=None, stat='identity', position='identity',
       na_rm=False, inherit_aes=True, show_legend=None)
    table1 = pd.read_csv(inpData1, sep="\t")

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
    except:
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
    return jsonify(
        message='success',
        # data=baseUrl+url_for('static', filename=f_name),
        data=baseUrl + url_for('static', filename=timeStr+'.png'),
    )
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


# if __name__ == '__main__':
#     app.run(debug=True)


