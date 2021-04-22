from http.server import BaseHTTPRequestHandler, HTTPServer
import sketch_utils
import sketch_detect
import random
import json
import ndjson
#from sketch_detect import detect


""" The HTTP request handler """
class RequestHandler(BaseHTTPRequestHandler):

    def _send_cors_headers(self):
        """ Sets headers required for CORS """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")


    def send_dict_response(self, d):
        """ Sends a dictionary (JSON) back to the client """
        self.wfile.write(bytes(json.dumps(d), "utf8"))


    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()


    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

        #   Meant only for responding?  Cannot figure out how to read in input data.

        response = {}
        response["status"] = "GET SUCCESS"
        self.send_dict_response(response)


    def do_POST(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        dataLength = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(dataLength).decode("UTF-8"))

        # If quickdraw fetch request, load a locally stored ndjson file into a json object and send it in response.
        # Start with a single sketch for now.  Also, reformat to match paper.js path json export format (collapse x and y arrays into single array of [x, y] tuples).
        if data["request"] == "fetchSketch":
            response = sketch_utils.fetchSketch(data["sketchCat"])
            self.send_dict_response(response)
        elif data["request"] == "createSketchSequence":
            sketch_utils.createSketchSequence(data)
        elif data["request"] == "predictSketch":
            response = sketch_detect.predict_sketch(data["sketchSequence"])
            self.send_dict_response(response)

        print("POST Complete.")


print("Starting server")
httpd = HTTPServer(("127.0.0.1", 8000), RequestHandler)
print("Hosting server on port 8000")
httpd.serve_forever()