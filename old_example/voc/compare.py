#!/usr/bin/env python
import sys
import os
import cv2
from math import sqrt
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


ERR_MSG = 'Creation of box with left top vertex {} and right bottum vertex {} is invalid'

class Box(object):
    def __init__(self, left_top, right_bot):
	if any(map(lambda x,y: x > y, left_top, right_bot)):
	    raise ValueError(ERR_MSG.format(left_top, right_bot))
	self.vertex = ((left_top, (right_bot[0], left_top[1])), ((left_top[0], right_bot[1]), right_bot))
    
    @property
    def left_top(self):
	return self.vertex[0][0]

    @property
    def right_bot(self):
        return self.vertex[1][1]

    @property
    def area(self):
	width = self.vertex[0][1][0] - self.vertex[0][0][0]
        height = self.vertex[1][0][1] - self.vertex[0][0][1]
	return width * height

    def intersection(self, box):
	try:
	    inter = Box(tuple(map(max, zip(self.left_top, box.left_top))), tuple(map(min, zip(self.right_bot, box.right_bot))))
	except ValueError:
	    inter = Box((0,0), (0,0))
	return inter

def boxes_generator(filename):
    with open(filename, 'r') as handle:
	group = []
	index = 0
	for line in handle.readlines():
	    frame_id, x1, y1, x2, y2, _ = map(int, line.split(' ')) 
	    if not index:
		index = frame_id
	    elif index != frame_id:
		yield index, tuple(group)
		group = []
		index = frame_id
            group.append(((x1, y1), (x2 ,y2)))


def compare_result(file1, file2, threshold=0.5):
    threshold= float(threshold)
    difference = []
    boxes = []
    box_gen1 = boxes_generator(file1)
    box_gen2 = boxes_generator(file2)
    
    current_frame = 1

    group1 = next(box_gen1, None)
    group2 = next(box_gen2, None)

    f = lambda x: {'correct': x[0], 'total': x[1], 'recall': float(x[0]) / x[1] if x[1] else x[0] == x[1], 'precision': float(x[0]) / x[2] if x[2] else x[0] == x[2], 'diff': abs(x[2] - x[1])}
    while True:
	if group1 and group1[0] < current_frame:
	    group1 = next(box_gen1, None)
	    
	if group2 and group2[0] < current_frame:
            group2 = next(box_gen2, None)

	if group1 is None and group2 is None:
	    return difference, boxes
	
	if not group1 or not group2:
	    box1 = group1[1] if group1 else []
	    box2 = group2[1] if group2 else []
	    difference.append(f((0, max(map(len, (box1, box2))), abs(len(box1) - len(box2)))))
	else:
	    if group1[0] > current_frame:
		box1 = []
	    else:
		box1 = group1[1]
	    if group2[0] > current_frame:
		box2 = []
	    else:
		box2 = group2[1]
	    difference.append(f((compare(box1, box2, threshold))))
	boxes.append((box1, box2))
	current_frame += 1
	
def compare(group1, group2, threshold):
    avg_precision = 0
    total = 0
    matched = set()
    for box1 in group1:
	max_iou = 0
	box_id = -1
	for i, box2 in enumerate(group2):
	    iou = IoU(Box(*box1), Box(*box2))
	    if iou >= threshold and max_iou < iou:
		max_iou = iou
		box_id = i
	total += 1
	if box_id != -1 and box_id not in matched:
	    matched.add(box_id)
	
    return len(matched), total, max(map(len,(group2, group1)))


def IoU(box1, box2):
    itersec = box1.intersection(box2)
    return float(itersec.area) / (box1.area + box2.area - itersec.area)

def simple_analysis(result):
    print('+++++++++++++++++++++++General Informations+++++++++++++++++++++++++++++')
    print('Total number of frames: {}'.format(len(result)))
    print('Frame wise Avg Recall: {}'.format(float(sum(map(lambda x: x['correct'],result))) / sum(map(lambda x: x['total'], result))))
    print('Frame wise Avg Precision: {}'.format(float(sum(map(lambda x: x['precision'],result))) / len(result)))
    print('{}% of the frames have same detection result'.format(float(sum(map(lambda x: x['recall'] >= 1.0 and x['precision'] >= 1.0 , result))) / len(result) * 100))
    with open(sys.argv[1], 'r') as f:
        for i, l in enumerate(f):
            pass
    print('Average bboxes in frame: {}'.format(i / float(len(result))))
    print('========================================================')


def analysis_continue_frames(result, threshold=1, miss_threshold=1, thresholdt=0.8, sqn_length = 1):
    cnt = 0
    mis_cnt = miss_threshold
    res = []
    bitarray = []
    for frame in result:
	if frame['recall'] >= threshold and frame['precision'] >= threshold:
	    cnt += 1
	    bitarray.append(True)
	elif cnt!= 0 and frame['recall'] >= thresholdt and frame['precision'] >= thresholdt:
	    mis_cnt -= 1
	    if mis_cnt < 0:
		mis_cnt = miss_threshold
		if cnt != 0:
		    res.append(cnt)
		    cnt = 0
	    else:
		cnt += 1
	    bitarray.append(True)
	else:
	    mis_cnt = miss_threshold
            if cnt != 0:
                res.append(cnt)
                cnt = 0
	    bitarray.append(False)

    mean = sum(res)/float(len(res)) 
    var = sum([d ** 2 for d in [x - mean for x in res]]) / len(res)
    ctr = Counter(res)
    grp_cnt = [0,0,0]
    thresh = [10, 30, 70]
    for i in ctr:
	if i in range(thresh[0]):
	    grp_cnt[0] += i * ctr[i]
	elif i in range(thresh[0], thresh[1]):
	    grp_cnt[1] += i * ctr[i]
	else:
	    grp_cnt[2] += i * ctr[i]

    print('+++++++++++++++++++++++++Distribution++++++++++++++++++++++++++++++++++')
    print('Miss Tolerance: {}, Threshold1 = {}, Threshold2 = {}'.format(miss_threshold, threshold, thresholdt))
    print('Fragments: {}'.format(len(res)))
    print('Average # of Contiune Frames share the same result: {}'.format(mean))
    print('Standard Diviation: {}'.format(sqrt(var)))
    print('The Frequency of Contiune Frames share the same result: {}'.format(ctr))
    print(grp_cnt)
    i = 0
    correct = 0
    speed_up = 0
    speed = [1/2.3, 1/6.2]
    while i < len(bitarray):
	mask = [bitarray[i]] * sqn_length
	zipped = zip(bitarray[i:i+sqn_length], mask)
	correct += [not (a == False and b == True) for a, b in zipped].count(True)
	i += sqn_length
	speed_up += speed[mask[0]] * len(zipped) + speed[1]*2*True
    ori_speed = len(bitarray) * speed[0]
    speedup_fact = (ori_speed - speed_up) / float(ori_speed)
    print('+++++++++++++++++++++++++++++Skip Simulation++++++++++++++++++++++++++++++++++++++++')
    print('If do classifcation every {} frames, {}({}%) frames will get lower accurancy out of {} frames'.format(sqn_length, len(bitarray) - correct, 100*(len(bitarray) - correct)/len(bitarray), len(bitarray)))
    print('The original speed = {}, after speed up = {}, speeded up by {}%'.format(ori_speed, speed_up, speedup_fact * 100))
    print('==============================================================================')


def draw_counter(ctr):
    labels, values = zip(*ctr.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes+width*0.5, labels)
    plt.show()

def output_frames_with_boxes(results, boxes, video, outdir, rng=None):
    video_path = os.path.abspath(video)
    outdir_path = os.path.abspath(outdir)
    index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    vidcap = cv2.VideoCapture(video_path)
    success = True
    while success:
	if (index + 1) % 100 == 0 or index + 1 == len(results):
	    print('Processing {}/{} frames'.format(index + 1, len(results)))
	group1, group2 = boxes[index]
	result = results[index]
	success,image = vidcap.read()
	for box in group1:
	    cv2.rectangle(image, box[0], box[1], (255,0,0), 5)
	for box in group2:
	    cv2.rectangle(image, box[0], box[1], (0,255,0), 5)
	if result['recall'] == 1 and result['precision'] == 1:
            cv2.putText(image, 'SAME', (200, 200), font, 3,(0,0,255),2)
            filename = 'frame%d_same.jpg'
	    cv2.imwrite(os.path.join(outdir_path, filename % index), image)
        else:
            filename = 'frame%d.jpg'
	    cv2.imwrite(os.path.join(outdir_path, filename % index), image)
	#cv2.imwrite(os.path.join(outdir_path, filename % index), image)
	index += 1
	if rng and index >= rng:
	    break

if __name__ == '__main__':
    result, boxes = compare_result(*sys.argv[1:3])
    simple_analysis(result)
    th2 = .8
    th1 = 1
    if len(sys.argv) > 3:
	output_frames_with_boxes(result, boxes, sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
    else:
        for skip in range(1, 61, 1):
    	    analysis_continue_frames(result, th1, 0, 1, skip)
