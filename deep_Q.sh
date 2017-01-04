#!/bin/bash
scp yilundu@visiongpu19.csail.mit.edu:/data/vision/billf/object-properties/yilundu/space_invaders/saved.h5 saved.h5
python deep_Q.py
