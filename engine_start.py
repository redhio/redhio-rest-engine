#
# Redhio Rest Engine
#
from flask import Flask
from flask_restful import request, reqparse, abort, Api, Resource
import requests

from flask_jsonpify import jsonify
from flask import send_file

import json
#import jsonpickle
from werkzeug.utils import secure_filename
from flask import Response, render_template, redirect, make_response
from flask_cors import CORS
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
from numpy import array
import cv2
import imutils
# sudo pip3 install flask_cors, flask_jsonpify, flask_restful, flask

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/service/*": {"origins": "*"}})
cors = CORS(app, resources={r"/api/predict/*": {"origins": "*"}})
cors = CORS(app, resources={r"/api/classify/*": {"origins": "*"}})

SERVICES = {
    'build': {'1.build': 'Build a ML model!'},
    'train': {'2.train': 'Train new elements or batches'},
    'analyse': {'3.analyse': 'Analyse the training results'},
    'deploy': {'4.deploy': 'Deploy a model'},
    'predict': {'5.predict': 'Predict ML Features'},
    'learn': {'6.learn': 'Learn from user feedback'},
    'services': {'100.services': 'Services List API'},
    'service': {'100.service': 'Service/<service_id> API'},
    'api': {'api/classify': 'Image Classification-Smiles'},
}

R_PRED = [
{ "confidence" : 0.5003, "label" : "Smiling" },
{ "confidence" : 0.4035, "label" : "Not Smiling" }
]

def abort_if_service_doesnt_exist(service_id):
    if service_id not in SERVICES:
        abort(404, message="Service {} doesn't exist".format(service_id))

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def smile_detector(in_file):
    
    in_frame = cv2.imread('uploaded_image.png') 
    cv2.imwrite('uploaded_image.jpg',in_frame)
    #print(in_file,frame)
    print('smile')
    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(in_frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()   
    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the face bounding boxes
    label = {'neutraling'}
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        
        # determine the probabilities of both "smiling" and "not
        # smiling", then set the label accordingly
        prediction ={'szmiling','notsmiling'}
        #prediction = model.predict(roi)[0]
        #with graph.as_default():
        #    labels = model.predict(roi)[0]
        #with tf.Session(graph=fgraph) as sess:
        #    y_out = sess.run(y,roi)
        
        x_flat = array(roi[0]).reshape(1, 28,28,1)
        
        
        #y_out = persistent_sess.run(y, feed_dict={
        #    x: x_flat
            # x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
        #    })
        #print(prediction)
        (notSmiling, smiling) = prediction
        label = "Smiling" if smiling > notSmiling else "Not Smiling" 
        print(prediction,label)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
            (0, 0, 255), 2)
    cv2.imwrite('uploaded_frame.png',frame)
    cv2.imwrite('uploaded_frameClone.png',frameClone)
    #K.clear_session()
    #return(label,'uploaded_frameClone.png')
    return(label,frameClone)

###########################################################################
# Predict
# shows a single prediction item and lets you retrain a service itemclass 
class Predict(Resource):
    def get(self):      
        return Response(render_template('./predict.html'),mimetype='text/html') 
    def post(self):
        args = parser.parse_args()
        data = request.data
        print(request,args)
        # convert string of image data to uint8 and decode image
        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('uploaded_image.png',img)
        
        #prediction,returnfile = smile_detector(data)
        #send_file('uploaded_frameClone.png' , mimetype='image/png')
        
        #return R_PRED, 201
        #return send_file(img,attachment_filename='uploaded_frameClone.png' ,mimetype='image/png')
        #""" post image and return the response """
        # prepare headers for http request
        headers = {'content-type': 'image/png'}        
        img_file = 'uploaded_frameClone.png'
        
        #response = make_response()
        #response.data = cv2.imread('uploaded_image.png')
        #response.headers = headers
        #return response, 201
        #return Response(response=R_PRED, status=201, data=img, headers=headers, mimetype="application/json",content_type=None)
        # build a response dict to send back to client
        #response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
        # encode response using jsonpickle
        #response_pickled = jsonpickle.encode(response)
        #return Response(response=response_pickled, status=200, mimetype="application/json")  
        
        res = send_file(img,
                            mimetype='image/png',
                            # mimetype='application/octet-stream',
                            #as_attachment=True,
                            attachment_filename='image.png')
        print(res)
        return R_PRED, 201
  
    
class Classify(Resource):
    def get(self):      
        return Response(render_template('./index.html'),mimetype='text/html') 
    def post(self):
        args = parser.parse_args()
        imgdata = request.data
        print(request,args)
        # convert string of image data to uint8
        nparr = np.fromstring(imgdata, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('uploaded_image.png',img)
        prediction,returnfile = smile_detector(imgdata)
	
        send_file(returnfile, mimetype='image/png')

        return R_PRED, 201
    
#    
# shows a single service item and lets you delete a service item
class Service(Resource):

    def get(self, service_id):
        abort_if_service_doesnt_exist(service_id)
        return SERVICES[service_id], 200

    def delete(self, service_id):
        abort_if_service_doesnt_exist(service_id)
        del SERVICES[service_id]
        return '', 204

    def put(self, service_id):
        args = parser.parse_args()
        service = {'service': args['service']}
        SERVICES[service_id] = service
        return service, 201


# ServiceList
# shows a list of all services, and lets you POST to add new tasks
class ServiceList(Resource):
    def get(self):
        return SERVICES

    def post(self):
        args = parser.parse_args()
        service_id = int(max(SERVICES.keys()).lstrip('service')) + 1
        service_id = 'service%i' % service_id
        SERVICES[service_id] = {'service': args['service']}
        return SERVICES[service_id], 201

class Logger(Resource):
	def log():
		g.uuid = uuid.uuid1().hex
		req_data = save_request(g.uuid, request)
		resp = Response(json.dumps(req_data, indent=4), mimetype='application/json')
		resp.set_cookie('cookie-name', value='cookie-value')
		return resp
##
## Actually setup the Api resource routing here
##
api.add_resource(Predict, '/api/predict')
api.add_resource(Classify, '/api/classify')
api.add_resource(Service, '/service/<service_id>')
api.add_resource(ServiceList, '/services')
api.add_resource(Logger, '/log')

if __name__ == '__main__':
    #app.run(debug=True)
    parser = reqparse.RequestParser()
    #parser.add_argument('service_id')
    #parser.add_argument('file',location='files')
    #parser.add_argument('data')
    ###########################################################################
    cascade='./haarcascade_frontalface_default.xml'
    model_store='./output/lenet.hdf5'
    frozen_model='./output/LeNet.pb'
    # load the face detector cascade and smile detector CNN
    detector = cv2.CascadeClassifier(cascade)
    model = load_model(model_store)
    graph = tf.get_default_graph()
    model._make_predict_function()
    print('testing graph:', model.predict(np.zeros((0, 28, 28, 1))))
    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    fgraph = load_graph(frozen_model)
    # We can verify that we can access the list of operations in the graph
    for op in fgraph.get_operations():
        print(op.name)
    x = fgraph.get_tensor_by_name('prefix/conv2d_3_input_1:0')
    y = fgraph.get_tensor_by_name('prefix/output0:0')
    gpu_memory=5
    print('Starting Session, setting the GPU memory usage to %f' % gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=fgraph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################
    print('Starting the API')

    
    app.run(debug=True,host='0.0.0.0', port=8000)
    cv2.destroyAllWindows()  
    camera.release()
    