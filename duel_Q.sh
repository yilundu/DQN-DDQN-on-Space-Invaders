#!/bin/bash
scp yilundu@visiongpu19.csail.mit.edu:/data/vision/billf/object-properties/yilundu/space_invaders/duel_saved.h5 duel_saved.h5
python duel_Q.py
