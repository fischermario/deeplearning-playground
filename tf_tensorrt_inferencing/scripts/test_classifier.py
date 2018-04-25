import os
import subprocess

import re

import sys
sys.path.append("scripts")
from conf import config

CLASSIFIER_EXE_PATH = 'build/examples/classify_image/classify_image'

def executeClassifier(image_filename, engine_filename, labels_filename, input_name, output_name):
    args = [
        image_filename,
        engine_filename,
        labels_filename,
        input_name,
        output_name,
        "vgg"
    ]
    
    proc = subprocess.Popen([CLASSIFIER_EXE_PATH] + args,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    output = ""

    for line in proc.stdout:
        output += line.decode('UTF-8')
    
    #for line in proc.stderr:
    #    print(line.decode('UTF-8'), end='')

    proc.wait()

    result = [-1, image_filename.split(os.path.sep)[-1]]

    #print(output)

    match = re.search(r"indices are: (\d+?).+?0\. (.+?)\n.+?'\2' probability: (.+?)\n", output, re.S)
    
    if match:
        result[0] = 0
        result += [int(match.group(1)), float(match.group(3)), match.group(2)]

    return result


if __name__ == '__main__':
    plan = str(config["engine_save_dir"]) + "keras_vgg19_b" + str(config["inference_batch_size"]) + "_"+ str(config["precision"]) + ".engine"

    for root, _, files in os.walk(config["test_image_path"]):
        for filename in sorted(files):
            fn_with_path = os.path.join(root, filename)
            if os.path.isfile(fn_with_path):
                result = executeClassifier(fn_with_path, plan, config['labels_file'], config['input_layer'], config['out_layer'])
                if result[0] == 0:
                    print(result[1])
                    print("{}: {:.2f}%".format(result[4], result[3] * 100))

