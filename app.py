from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os.path
import re
import sys
import tarfile
 
import numpy as np
from six.moves import urllib
import tensorflow as tf
 
from flask import Flask, request, Response, jsonify
from werkzeug import secure_filename

app = Flask(__name__)

sess = None
node_lookup = None

 # Creates graph from saved graph_def.pb.
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

print("Model loaded")

label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]
print("labels loaded")
	
sess = tf.Session()
print("Tensorflow session ready")


# import default command line flags from TensorFlow
FLAGS = tf.app.flags.FLAGS
 
# define directory that the model is stored in (default is the current directory)
tf.app.flags.DEFINE_string(
  'model_dir', '.',
  """Path to output_graph.pb, """
  """output_labels.txt""")
 
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

@app.route('/', methods=['GET', 'POST'])
def welcome():
	return "welcome to deepAlpr API landing page"
 
# Classificaiton endpoint
@app.route("/classify", methods=["POST"])
def classify():
  image= request.files['file']
  image.save(secure_filename(image.filename))
 # setup_app()
  predictions = dict(run_inference_on_image(image.filename))
  print(predictions)
  return jsonify(predictions=predictions)
 
# The following code performs the recognition, and is derived from the examples
# provided in the Tensorflow package
# ==============================================================================
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
 
"""Simple image classification with Inception.
 
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
 
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
 
Change the --image_file argument to any jpg image to compute a
classification of that image.
 
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
 
https://tensorflow.org/tutorials/image_recognition/
"""
 
 
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""
 
  def __init__(self, label_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'retrained_labels.txt')
 
   # self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
 
  def load(self, label_lookup_path):
    """Loads a human readable English name for each softmax node.
 
    Args:
      label_lookup_path: string UID to integer node ID.
 
    Returns:
      dict from integer node ID to human-readable string.
    """
 
    node_id_to_name = {}
 
    label_file = open(label_lookup_path)
    i = 0
 
    # labels are ordered from 0 to N in the lookup file
 
    for line in label_file:
      node_id_to_name[i] = line.strip()
      i = i + 1
 
    return node_id_to_name
 
  # return the friendly name for the given node_id
  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]
 
 
def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
    FLAGS.model_dir, 'retrained_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
 

 
def run_inference_on_image(image_data):
  """Runs inference on an image.
 
  Args:
    image_data: Image data.
 
  Returns:
    Nothing
  """
  image_data_read = tf.gfile.FastGFile(image_data, 'rb').read()
  # Runs the softmax tensor by feeding the image_data as input to the graph.
  softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
  predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data_read})
  #predictions = np.squeeze(predictions)
  # sort the predictions
  top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
  for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
  # map to the friendly names and return the tuples
  return [{'done'}]
 

def setup_app():
   create_graph()
   print("Model loaded")

   node_lookup = NodeLookup()
   print("Node lookup loaded")

   sess = tf.Session()
   print("Tensorflow session ready")

   print("Launching web application...")
  # app.run(debug=True)


 
if __name__ == '__main__':
  print("Launching web application...")
  app.run(debug=True)
  

