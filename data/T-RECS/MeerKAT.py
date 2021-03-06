#! /usr/bin/python3
# -*- coding: utf-8 -*-

class MeerKATarray(object):
    def __init__(self):
        import numpy as np
        from astropy.coordinates import EarthLocation
        import astropy.units as u
        self.Array=np.array([[ 5109271.49735416,  5109284.85407754,  5109272.1993435 ,
         5109294.92858499,  5109291.9021494 ,  5109233.71782573,
         5109209.26348941,  5109148.3563611 ,  5109180.03524117,
         5109093.55655612,  5109170.18055871,  5109142.24732406,
         5109095.43320709,  5109130.11047997,  5109186.74678517,
         5109174.26788279,  5109240.95388574,  5109212.2308251 ,
         5109170.17197292,  5109190.13422423,  5109319.27502221,
         5109501.27803052,  5109415.83816637,  5109563.6943163 ,
         5109409.88085549,  5109340.50686944,  5109343.51732174,
         5109339.74820941,  5109357.50532698,  5109320.53511894,
         5109280.81866453,  5109561.41124047,  5109223.0447991 ,
         5109141.20538522,  5109088.62895199,  5109012.51005451,
         5109021.19439314,  5109040.86390684,  5109158.68572906,
         5109280.35702671,  5109533.61299471,  5109972.69938675,
         5110157.09530499,  5110723.7009419 ,  5109331.7459565 ,
         5111655.26092217,  5110888.06656438,  5109713.65348687,
         5109311.35148968,  5109039.4227322 ,  5108748.65570024,
         5108814.45202929,  5108974.66330238,  5109003.20020234,
         5110793.52095214,  5109608.66590919,  5108382.61808825,
         5107254.01347188,  5108278.55916154,  5108713.98241022,
         5109748.52632071],
       [ 2006808.89302781,  2006824.22172353,  2006783.54604995,
         2006755.50985406,  2006692.96802726,  2006783.34519269,
         2006697.07228302,  2006668.92154865,  2006816.61011285,
         2006842.5269473 ,  2006868.23609545,  2006917.43739644,
         2007003.01945823,  2007063.78075524,  2007010.79288481,
         2007089.17193214,  2007020.24474701,  2006908.08257804,
         2006961.46137943,  2006890.0451634 ,  2006518.56060045,
         2006507.28233036,  2006528.18913299,  2006555.38709425,
         2006765.75924307,  2006888.36185406,  2006791.07431684,
         2006749.21628934,  2007035.57284496,  2007101.93689326,
         2007317.01340668,  2007555.5082679 ,  2007183.16215354,
         2007181.46662131,  2007163.07668579,  2007124.61491175,
         2006948.62994241,  2006698.32802902,  2006464.53114915,
         2006432.6365746 ,  2006244.02924612,  2006130.37573833,
         2005196.44300058,  2005811.69710478,  2006220.13139806,
         2004739.74954867,  2003578.58732047,  2004786.46861502,
         2005919.93511349,  2006089.30840302,  2006622.47104818,
         2007575.08416726,  2007992.33987636,  2008429.66935399,
         2007732.1493962 ,  2009964.63970196,  2010429.23248703,
         2009699.35721797,  2006410.13690606,  2005051.01654913,
         2003331.23203868],
       [-3239130.73614072, -3239100.12646042, -3239145.33004168,
        -3239126.26624966, -3239169.94641344, -3239206.81520042,
        -3239299.36668257, -3239413.45218346, -3239272.32242145,
        -3239393.92388052, -3239256.01206258, -3239270.0825518 ,
        -3239292.08731167, -3239199.22441129, -3239141.50848633,
        -3239112.93487932, -3239049.21915551, -3239164.38124624,
        -3239198.49211401, -3239210.78638671, -3239233.61957711,
        -3238950.54476697, -3239073.85646735, -3238821.56314329,
        -3238936.95141411, -3238971.92750107, -3239026.97172989,
        -3239058.76457829, -3238853.87633838, -3238871.33463575,
        -3238800.99879004, -3238207.17422081, -3238977.04746739,
        -3239108.90628107, -3239203.63040132, -3239349.23577353,
        -3239443.49706673, -3239566.50339235, -3239522.08840298,
        -3239348.22709537, -3239061.14460898, -3238431.52176684,
        -3238718.84306904, -3237438.29057988, -3239396.7638735 ,
        -3236633.10524755, -3238574.25451153, -3239676.24008691,
        -3239613.47504072, -3239941.07331375, -3240077.32054145,
        -3239384.66757419, -3238870.28669984, -3238550.47820352,
        -3236139.97163011, -3236636.20942028, -3238301.70200615,
        -3240542.58734053, -3240956.88531359, -3241111.82913216,
        -3240538.85373571]])


        self.Loc=EarthLocation(
        lat=-30.83 * u.deg,
        lon=21.33 * u.deg,
        height=1195. * u.m
    )