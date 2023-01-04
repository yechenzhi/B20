import json
import os
from copy import deepcopy

if __name__ == '__main__':
    trainB20 = json.load(open('./valB20v2.json','r'))
    anns_c1 = []
    for ann in trainB20['annotations']:   
        ann_t = deepcopy(ann)
        ann_t['category_id'] = 0
        anns_c1.append(ann_t)
    # import pdb; pdb.set_trace()
    cats_c1 = {}
    cats_c1.update(id=0,name='fg')

    trainB20C1 = {}
    trainB20C1.update(images=trainB20['images'],annotations=anns_c1,categories=[cats_c1])
    
    with open('valB20C1.json','w') as f: 
        json.dump(trainB20C1,f)