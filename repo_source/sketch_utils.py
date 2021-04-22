import ndjson
import random
import base64
import os

def fetchSketch(sketchCat):
    print("Loading sketch ndjson.")
    with open('sketch_data/quickdraw_preprocessed_data/full_simplified_' + sketchCat + '.ndjson') as f:
        sketches = ndjson.load(f)

    #response = sketches[0]["drawing"]
    response = []
    strokeHolder = []
    sketchHolder = []
    #   Some googling suggests that only 1k training instances per class is needed.
    # missingSketches = []
    # for i in range(1000):
    #     if not os.path.isdir(f"sketch_data/converted_data/{sketchCat}/{sketchCat}_{i+1}/"):
    #         missingSketches.append(i)

    numSketches = 2000
    #whichSketch = random.randrange(len(sketches))
    #whichSketch = 89154
    #print(f"Which sketch: {whichSketch}")

    print("Reformatting sketches.")
    # for l in range(len(missingSketches)):
    #     i = missingSketches[l]
    for i in range(numSketches):
        #   sketches[which drawing][drawing attribute that contains coords][which stroke][x, y, time][which point]
        for j in range(len(sketches[i]["drawing"])):            #   Each stroke.
            for k in range(len(sketches[i]["drawing"][j][0])):     #   Each coordinate ([0] is x list and [1] is y list.
                strokeHolder.append([sketches[i]["drawing"][j][0][k], sketches[i]["drawing"][j][1][k]])
            sketchHolder.append(strokeHolder)
            strokeHolder = []
        response.append(sketchHolder)
        sketchHolder = []

    return response


def createSketchSequence(data):
    sketchSequence = data["sketchSequence"]
    category = data["sketchCat"]
    sketchIndex = data["sketchIndex"]

    sequencePath = f"sketch_data/converted_data/{category}/{category}_{sketchIndex+1}"
    os.mkdir(sequencePath)

    for i in range(len(sketchSequence)):
        with open(f"{sequencePath}/{i+1}.png", "wb") as f:
            f.write(base64.b64decode(sketchSequence[i]))



# sketchCat = "tree"
# cats = ""
# for i in range(2000):
#         if not os.path.isdir(f"sketch_data/converted_data/{sketchCat}/{sketchCat}_{i+1}/"):
#             cats = f"{cats} {i},"
#
# print(cats)